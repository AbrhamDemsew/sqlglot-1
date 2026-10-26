import collections
import itertools
import math

from sqlglot import exp, planner, tokens
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import ExecuteError
from sqlglot.executor.context import Context
from sqlglot.executor.env import ENV
from sqlglot.executor.table import RowReader, Table
from sqlglot.generators.python import PythonGenerator
from sqlglot.optimizer.scope import build_scope

SUBQUERY_NODES = (exp.Subquery, exp.Exists, exp.All, exp.Any)


class PythonExecutor:
    def __init__(self, env=None, tables=None):
        self.generator = Python().generator(identify=True, comments=False)
        self.env = {**ENV, **(env or {})}
        self.tables = tables or {}
        self._subquery_plans = {}
        self._plan_names_by_sql = {}
        self._ctes = None
        self._outer_scope = None
        self.env.update(
            SUBQUERY_COMPARISON=self._subquery_comparison,
            SUBQUERY_EXISTS=self._subquery_exists,
            SUBQUERY_SCALAR=self._subquery_scalar,
        )

    def execute(self, plan, outer_scope=None):
        ctes, scope = self._ctes, self._outer_scope
        self._ctes, self._outer_scope = plan.ctes, outer_scope
        try:
            return self._execute(plan)
        finally:
            self._ctes, self._outer_scope = ctes, scope

    def _execute(self, plan):
        finished = set()
        queue = set(plan.leaves)
        contexts = {}

        while queue:
            node = queue.pop()
            try:
                context = self.context(
                    {
                        name: table
                        for dep in node.dependencies
                        for name, table in contexts[dep].tables.items()
                    }
                )

                if isinstance(node, planner.Scan):
                    contexts[node] = self.scan(node, context)
                elif isinstance(node, planner.Aggregate):
                    contexts[node] = self.aggregate(node, context)
                elif isinstance(node, planner.Join):
                    contexts[node] = self.join(node, context)
                elif isinstance(node, planner.Sort):
                    contexts[node] = self.sort(node, context)
                elif isinstance(node, planner.Window):
                    contexts[node] = self.window(node, context)
                elif isinstance(node, planner.SetOperation):
                    contexts[node] = self.set_operation(node, context)
                else:
                    raise NotImplementedError

                if node.offset:
                    table = contexts[node].tables[node.name]
                    table.rows = table.rows[node.offset :]

                finished.add(node)

                for dep in node.dependents:
                    if all(d in contexts for d in dep.dependencies):
                        queue.add(dep)

                for dep in node.dependencies:
                    if all(d in finished for d in dep.dependents):
                        contexts.pop(dep)
            except Exception as e:
                raise ExecuteError(f"Step '{node.id}' failed: {e}") from e

        root = plan.root
        return contexts[root].tables[root.name]

    def generate(self, expression):
        """Convert a SQL expression into literal Python code and compile it into bytecode."""
        if not expression:
            return None

        expression = self._replace_subqueries(expression)
        sql = self.generator.generate(expression)
        return compile(sql, sql, "eval", optimize=2)

    def _replace_subqueries(self, expression):
        if not expression.find(*SUBQUERY_NODES):
            return expression

        expression = expression.copy()

        while True:
            subquery = expression.find(*SUBQUERY_NODES)

            if subquery is None:
                return expression

            target, replacement = self._compile_subquery(subquery)

            if target is expression:
                expression = replacement
            else:
                target.replace(replacement)

    def _compile_subquery(self, subquery):
        query = subquery.this.unnest()
        scope = build_scope(query)

        outer_columns = list(scope.external_columns if scope else [])

        plan = self._register_subquery(query)
        parent = subquery.parent

        if isinstance(subquery, exp.Exists):
            return subquery, exp.func(
                "SUBQUERY_EXISTS", plan, exp.var("scope"), *outer_columns, copy=False
            )

        if len(query.selects) != 1:
            raise ExecuteError(
                f"Subquery used as an expression returned {len(query.selects)} columns"
            )

        if isinstance(subquery, (exp.All, exp.Any)):
            return self._compile_quantified(parent, subquery.key.upper(), plan, outer_columns)

        if isinstance(parent, exp.In) and subquery is parent.args.get("query"):
            return self._compile_quantified(parent, "ANY", plan, outer_columns, op="EQ")

        return subquery, exp.func(
            "SUBQUERY_SCALAR", plan, exp.var("scope"), *outer_columns, copy=False
        )

    def _compile_quantified(self, comparison, quantifier, plan, outer_columns, op=None):
        if not isinstance(comparison, (exp.Binary, exp.In)):
            raise ExecuteError(f"Unsupported {quantifier} subquery: expected a comparison")

        return comparison, exp.func(
            "SUBQUERY_COMPARISON",
            comparison.this,
            plan,
            exp.var("scope"),
            exp.Literal.string(op or comparison.key.upper()),
            exp.Literal.string(quantifier),
            *outer_columns,
            copy=False,
        )

    def _register_subquery(self, query):
        if self._ctes is not None and not query.args.get("with_"):
            query.set("with_", self._ctes)

        sql = query.sql()
        name = self._plan_names_by_sql.get(sql)

        if name is None:
            name = self._plan_names_by_sql[sql] = f"_sq_{len(self._subquery_plans)}"
            self._subquery_plans[name] = (planner.Plan(query), {})

        return exp.Literal.string(name)

    def _subquery_table(self, plan_name, scope, args):
        plan, cache = self._subquery_plans[plan_name]

        try:
            return cache[args]
        except KeyError:
            pass
        except TypeError:  # an unhashable correlated value can't be memoized
            cache = None

        table = self.execute(plan, scope)

        if cache is not None:
            cache[args] = table

        return table

    def _subquery_exists(self, plan_name, scope, *args):
        return bool(self._subquery_table(plan_name, scope, args).rows)

    def _subquery_scalar(self, plan_name, scope, *args):
        rows = self._subquery_table(plan_name, scope, args).rows

        if len(rows) > 1:
            raise ExecuteError("More than one row returned by a subquery used as an expression")

        return rows[0][0] if rows else None

    def _subquery_comparison(self, value, plan_name, scope, op, quantifier, *args):
        compare = self.env[op]
        is_any = quantifier == "ANY"
        saw_null = False

        for row in self._subquery_table(plan_name, scope, args).rows:
            result = compare(value, row[0])

            if result is None:
                saw_null = True
            elif bool(result) is is_any:
                return is_any

        return None if saw_null else not is_any

    def generate_tuple(self, expressions):
        """Convert an array of SQL expressions into tuple of Python byte code."""
        if not expressions:
            return tuple()
        return tuple(self.generate(expression) for expression in expressions)

    def context(self, tables):
        return Context(tables, env=self.env, outer=self._outer_scope)

    def table(self, expressions):
        return Table(
            expression.alias_or_name if isinstance(expression, exp.Expr) else expression
            for expression in expressions
        )

    def scan(self, step, context):
        source = step.source

        if source and isinstance(source, exp.Expr):
            source = source.name or source.alias

        if source is None:
            context, table_iter = self.static()
        elif source in context:
            if not step.projections and not step.condition:
                return self.context({step.name: context.tables[source]})
            table_iter = context.table_iter(source)
        else:
            context, table_iter = self.scan_table(step)

        return self.context({step.name: self._project_and_filter(context, step, table_iter)})

    def _project_and_filter(self, context, step, table_iter):
        sink = self.table(step.projections if step.projections else context.columns)
        condition = self.generate(step.condition)
        projections = self.generate_tuple(step.projections)

        for reader in table_iter:
            if len(sink) >= step.offset + step.limit:
                break

            if condition and not context.eval(condition):
                continue

            if projections:
                sink.append(context.eval_tuple(projections))
            else:
                sink.append(reader.row)

        return sink

    def static(self):
        return self.context({}), [RowReader(())]

    def scan_table(self, step):
        table = self.tables.find(step.source)
        context = self.context({step.source.alias_or_name: table})
        return context, iter(table)

    def join(self, step, context):
        source = step.source_name

        source_table = context.tables[source]
        source_context = self.context({source: source_table})
        column_ranges = {source: range(0, len(source_table.columns))}

        for name, join in step.joins.items():
            table = context.tables[name]
            start = max(r.stop for r in column_ranges.values())
            column_ranges[name] = range(start, len(table.columns) + start)
            join_context = self.context({name: table})
            condition = self.generate(join["condition"])
            condition_context = (
                self.context(
                    {
                        name: Table(
                            source_context.columns + join_context.columns,
                            column_range=column_range,
                        )
                        for name, column_range in column_ranges.items()
                    }
                )
                if condition
                else None
            )

            if join.get("source_key"):
                table = self.hash_join(
                    join, source_context, join_context, condition, condition_context
                )
            else:
                table = self.nested_loop_join(
                    join, source_context, join_context, condition, condition_context
                )

            source_context = self.context(
                {
                    name: Table(table.columns, table.rows, column_range)
                    for name, column_range in column_ranges.items()
                }
            )
        if not step.condition and not step.projections:
            return source_context

        sink = self._project_and_filter(
            source_context,
            step,
            (reader for reader, _ in iter(source_context)),
        )

        if step.projections:
            return self.context({step.name: sink})
        else:
            return self.context(
                {
                    name: Table(table.columns, sink.rows, table.column_range)
                    for name, table in source_context.tables.items()
                }
            )

    @staticmethod
    def _join_matches(row, condition, condition_context):
        if not condition:
            return True

        condition_context.set_row(row)
        return condition_context.eval(condition) is True

    def nested_loop_join(self, join, source_context, join_context, condition, condition_context):
        table = Table(source_context.columns + join_context.columns)
        source_rows = source_context.table.rows
        join_rows = join_context.table.rows
        matched_source = set()
        matched_join = set()

        for source_index, source_row in enumerate(source_rows):
            for join_index, join_row in enumerate(join_rows):
                row = source_row + join_row
                if self._join_matches(row, condition, condition_context):
                    table.append(row)
                    matched_source.add(source_index)
                    matched_join.add(join_index)

        self._append_unmatched_join_rows(
            table, join, source_rows, join_rows, matched_source, matched_join
        )

        return table

    def hash_join(self, join, source_context, join_context, condition, condition_context):
        source_key = self.generate_tuple(join["source_key"])
        join_key = self.generate_tuple(join["join_key"])
        results = collections.defaultdict(lambda: ([], []))

        for index, (reader, ctx) in enumerate(source_context):
            key = ctx.eval_tuple(source_key)
            if all(value is not None for value in key):
                results[key][0].append((index, reader.row))
        for index, (reader, ctx) in enumerate(join_context):
            key = ctx.eval_tuple(join_key)
            if all(value is not None for value in key):
                results[key][1].append((index, reader.row))

        table = Table(source_context.columns + join_context.columns)
        matched_source = set()
        matched_join = set()

        for source_group, join_group in results.values():
            for (source_index, source_row), (join_index, join_row) in itertools.product(
                source_group, join_group
            ):
                row = source_row + join_row
                if self._join_matches(row, condition, condition_context):
                    table.append(row)
                    matched_source.add(source_index)
                    matched_join.add(join_index)

        self._append_unmatched_join_rows(
            table,
            join,
            source_context.table.rows,
            join_context.table.rows,
            matched_source,
            matched_join,
        )

        return table

    @staticmethod
    def _append_unmatched_join_rows(
        table, join, source_rows, join_rows, matched_source, matched_join
    ):
        side = join.get("side")
        if side in ("LEFT", "FULL"):
            join_nulls = (None,) * (len(table.columns) - len(source_rows[0]) if source_rows else 0)
            for index, row in enumerate(source_rows):
                if index not in matched_source:
                    table.append(row + join_nulls)

        if side in ("RIGHT", "FULL"):
            source_width = len(table.columns) - (len(join_rows[0]) if join_rows else 0)
            source_nulls = (None,) * source_width
            for index, row in enumerate(join_rows):
                if index not in matched_join:
                    table.append(source_nulls + row)

    def aggregate(self, step, context):
        group_by = self.generate_tuple(step.group.values())
        aggregations = self.generate_tuple(step.aggregations)
        operands = self.generate_tuple(step.operands)

        if operands:
            operand_table = Table(self.table(step.operands).columns)

            for reader, ctx in context:
                operand_table.append(ctx.eval_tuple(operands))

            for i, (a, b) in enumerate(zip(context.table.rows, operand_table.rows)):
                context.table.rows[i] = a + b

            width = len(context.columns)
            context.add_columns(*operand_table.columns)

            operand_table = Table(
                context.columns,
                context.table.rows,
                range(width, width + len(operand_table.columns)),
            )

            context = self.context(
                {
                    None: operand_table,
                    **context.tables,
                }
            )

        context.sort(group_by)

        group = None
        start = 0
        end = 1
        length = len(context.table)
        table = self.table(list(step.group) + step.aggregations)

        def add_row():
            table.append(group + context.eval_tuple(aggregations))

        if length:
            for i in range(length):
                context.set_index(i)
                key = context.eval_tuple(group_by)
                group = key if group is None else group
                end += 1
                if key != group:
                    context.set_range(start, end - 2)
                    add_row()
                    group = key
                    start = end - 2
                if not step.condition and len(table.rows) >= step.offset + step.limit:
                    break
                if i == length - 1:
                    context.set_range(start, end - 1)
                    add_row()
        elif step.limit > 0 and not group_by:
            context.set_range(0, 0)
            table.append(context.eval_tuple(aggregations))

        context = self.context({step.name: table, **{name: table for name in context.tables}})

        if step.projections or step.condition:
            return self.context(
                {step.name: self._project_and_filter(context, step, context.table_iter(step.name))}
            )
        return context

    def sort(self, step, context):
        projections = self.generate_tuple(step.projections)
        projection_columns = [p.alias_or_name for p in step.projections]
        all_columns = list(context.columns) + projection_columns
        sink = self.table(all_columns)
        for reader, ctx in context:
            sink.append(reader.row + ctx.eval_tuple(projections))

        sort_ctx = self.context(
            {
                None: sink,
                **{table: sink for table in context.tables},
            }
        )
        sort_ctx.sort(self.generate_tuple(step.key))

        if not math.isinf(step.limit):
            sort_ctx.table.rows = sort_ctx.table.rows[0 : step.offset + step.limit]

        rows = sort_ctx.table.rows

        if projection_columns:
            rows = [row[len(context.columns) : len(all_columns)] for row in rows]

        output = Table(projection_columns or context.columns, rows=rows)
        return self.context({step.name: output})

    def window(self, step, context):
        source_table = context.tables[step.source]
        rows = list(source_table.rows)
        source_cols = list(source_table.columns)

        window_results: dict[str, list] = {}
        for win_expr in step.windows:
            alias_name = win_expr.alias_or_name
            win_node = win_expr.find(exp.Window)
            if win_node is None:
                continue
            window_results[alias_name] = self._eval_window(
                win_node, rows, source_table, source_cols, context
            )

        new_cols = source_cols + list(window_results)
        out_table = Table(new_cols)
        for i, row in enumerate(rows):
            new_row = row + tuple(window_results[c][i] for c in window_results)
            out_table.append(new_row)

        if step.qualify is not None:
            qualify_expr = step.qualify.copy()
            extra_wins: dict[str, list] = {}
            for node in qualify_expr.walk():
                if isinstance(node, exp.Window):
                    tmp_col = f"__q_{len(extra_wins)}"
                    extra_wins[tmp_col] = self._eval_window(
                        node, rows, source_table, source_cols, context
                    )
                    node.replace(exp.column(tmp_col, step.name, quoted=True))

            all_cols = new_cols + list(extra_wins)
            aug_table = Table(all_cols)
            for i, row in enumerate(out_table.rows):
                extra_vals = tuple(extra_wins[c][i] for c in extra_wins)
                aug_table.append(row + extra_vals)

            qual_ctx = self.context({step.name: aug_table})
            qualify_code = self.generate(qualify_expr)
            kept_rows = []
            for reader, ctx in qual_ctx:
                if ctx.eval(qualify_code):
                    kept_rows.append(reader.row[: len(new_cols)])
            out_table = Table(new_cols, kept_rows)

        if step.projections:
            proj_ctx = self.context(
                {step.name: out_table, **{name: out_table for name in context.tables}}
            )
            proj_table = self.table(step.projections)
            proj_codes = self.generate_tuple(step.projections)
            for reader, ctx in proj_ctx:
                proj_table.append(ctx.eval_tuple(proj_codes))
            return self.context({step.name: proj_table})

        return self.context({step.name: out_table})

    def _eval_window(self, win_node, rows, source_table, source_cols, context):
        n = len(rows)
        func = win_node.this
        partition_exprs = win_node.args.get("partition_by") or []
        order_node = win_node.args.get("order")
        order_exprs = order_node.expressions if order_node else []
        spec = win_node.args.get("spec")

        def make_row_ctx(row):
            tbl = Table(source_cols, [row])
            return self.context({source_table.name or win_node.args.get("alias"): tbl,
                                  **{n: Table(source_cols, [row]) for n in context.tables}})

        def eval_expr_on_row(expr, row):
            ctx = self.context(
                {name: Table(source_cols, [row]) for name in context.tables}
            )
            ctx.set_index(0)
            code = self.generate(expr)
            return ctx.eval(code) if code else None

        def eval_col(expr, row_idx):
            row = rows[row_idx]
            tbl_name = next(iter(context.tables), None)
            if tbl_name is None:
                return None
            tbl = Table(source_cols, [row])
            ctx = self.context({tbl_name: tbl})
            ctx.set_index(0)
            code = self.generate(expr)
            return ctx.eval(code) if code else None

        def get_part_key(row_idx):
            row = rows[row_idx]
            tbl_name = next(iter(context.tables), None)
            if not partition_exprs or tbl_name is None:
                return ()
            tbl = Table(source_cols, [row])
            ctx = self.context({tbl_name: tbl})
            ctx.set_index(0)
            return tuple(
                ctx.eval(self.generate(e)) if self.generate(e) else None
                for e in partition_exprs
            )

        def get_order_vals(row_idx):
            row = rows[row_idx]
            tbl_name = next(iter(context.tables), None)
            if not order_exprs or tbl_name is None:
                return ()
            tbl = Table(source_cols, [row])
            ctx = self.context({tbl_name: tbl})
            ctx.set_index(0)
            vals = []
            for oe in order_exprs:
                code = self.generate(oe.this)
                vals.append(ctx.eval(code) if code else None)
            return tuple(vals)

        from sqlglot.executor.env import ordered as _ordered

        def sort_key_for(row_idx):
            row = rows[row_idx]
            tbl_name = next(iter(context.tables), None)
            if tbl_name is None:
                return ()
            tbl = Table(source_cols, [row])
            ctx = self.context({tbl_name: tbl})
            ctx.set_index(0)
            key_parts = []
            for oe in order_exprs:
                desc = bool(oe.args.get("desc"))
                nulls_first = bool(oe.args.get("nulls_first"))
                code = self.generate(oe.this)
                v = ctx.eval(code) if code else None
                key_parts.append(_ordered(v, desc, nulls_first))
            return tuple(key_parts)

        partitions: dict[tuple, list[int]] = {}
        for i in range(n):
            k = get_part_key(i)
            partitions.setdefault(k, []).append(i)

        results: list = [None] * n

        for part_indices in partitions.values():
            sorted_indices = sorted(part_indices, key=sort_key_for)
            order_vals_list = [get_order_vals(i) for i in sorted_indices]

            def frame_bounds(pos, sorted_part):
                if spec is None:
                    if order_exprs:
                        kind = "RANGE"
                        start_val, start_side = "UNBOUNDED", "PRECEDING"
                        end_val, end_side = "CURRENT ROW", None
                    else:
                        return 0, len(sorted_part) - 1
                else:
                    kind = spec.args.get("kind", "RANGE")
                    start_val = spec.args.get("start")
                    start_side = spec.args.get("start_side")
                    end_val = spec.args.get("end")
                    end_side = spec.args.get("end_side")

                ov = order_vals_list[pos]

                def resolve_bound(val, side, is_start):
                    if val == "UNBOUNDED":
                        return 0 if is_start else len(sorted_part) - 1
                    if val == "CURRENT ROW" or (val is None and side is None):
                        if kind == "ROWS":
                            return pos
                        lo, hi = pos, pos
                        while lo > 0 and order_vals_list[lo - 1] == ov:
                            lo -= 1
                        while hi < len(sorted_part) - 1 and order_vals_list[hi + 1] == ov:
                            hi += 1
                        return lo if is_start else hi
                    if kind == "ROWS":
                        offset = int(val.name) if hasattr(val, "name") else int(val)
                        if side == "PRECEDING":
                            return max(0, pos - offset)
                        return min(len(sorted_part) - 1, pos + offset)
                    numeric_offset = float(val.name) if hasattr(val, "name") else float(val)
                    if not ov or len(ov) != 1:
                        return 0 if is_start else len(sorted_part) - 1
                    cur = ov[0]
                    if cur is None:
                        lo2, hi2 = pos, pos
                        while lo2 > 0 and order_vals_list[lo2 - 1] == ov:
                            lo2 -= 1
                        while hi2 < len(sorted_part) - 1 and order_vals_list[hi2 + 1] == ov:
                            hi2 += 1
                        return lo2 if is_start else hi2
                    desc = bool(order_exprs[0].args.get("desc")) if order_exprs else False
                    if side == "PRECEDING":
                        bound_val = cur - numeric_offset if not desc else cur + numeric_offset
                    else:
                        bound_val = cur + numeric_offset if not desc else cur - numeric_offset
                    if side == "PRECEDING":
                        idx = 0
                        for j, ov2 in enumerate(order_vals_list):
                            v2 = ov2[0] if ov2 else None
                            if v2 is None:
                                continue
                            if (not desc and v2 >= bound_val) or (desc and v2 <= bound_val):
                                idx = j
                                break
                        else:
                            idx = len(sorted_part) - 1 if not is_start else 0
                        return idx if is_start else idx
                    else:
                        idx = len(sorted_part) - 1
                        for j in range(len(sorted_part) - 1, -1, -1):
                            ov2 = order_vals_list[j]
                            v2 = ov2[0] if ov2 else None
                            if v2 is None:
                                continue
                            if (not desc and v2 <= bound_val) or (desc and v2 >= bound_val):
                                idx = j
                                break
                        return idx

                s = resolve_bound(start_val, start_side, True)
                e = resolve_bound(end_val, end_side, False)
                return min(s, e), max(s, e)

            def get_frame_values(pos, col_expr, sorted_part):
                lo, hi = frame_bounds(pos, sorted_part)
                vals = []
                tbl_name = next(iter(context.tables), None)
                if tbl_name is None:
                    return vals
                for j in range(lo, hi + 1):
                    orig_idx = sorted_part[j]
                    row = rows[orig_idx]
                    tbl = Table(source_cols, [row])
                    ctx = self.context({tbl_name: tbl})
                    ctx.set_index(0)
                    code = self.generate(col_expr)
                    vals.append(ctx.eval(code) if code else None)
                return vals

            for pos, orig_idx in enumerate(sorted_indices):
                if isinstance(func, exp.RowNumber):
                    results[orig_idx] = pos + 1

                elif isinstance(func, exp.Rank):
                    ov = order_vals_list[pos]
                    count_before = sum(1 for j in range(pos) if order_vals_list[j] != ov)
                    results[orig_idx] = count_before + 1

                elif isinstance(func, exp.DenseRank):
                    ov = order_vals_list[pos]
                    seen_distinct: list = []
                    for j in range(pos):
                        prev = order_vals_list[j]
                        if prev != ov and prev not in seen_distinct:
                            seen_distinct.append(prev)
                    results[orig_idx] = len(seen_distinct) + 1

                elif isinstance(func, (exp.Lag, exp.Lead)):
                    offset_expr = func.args.get("offset")
                    default_expr = func.args.get("default")
                    offset = 1
                    if offset_expr is not None:
                        tbl_name = next(iter(context.tables), None)
                        if tbl_name:
                            row = rows[orig_idx]
                            tbl = Table(source_cols, [row])
                            ctx = self.context({tbl_name: tbl})
                            ctx.set_index(0)
                            code = self.generate(offset_expr)
                            offset = ctx.eval(code) if code else 1
                    default = None
                    if default_expr is not None:
                        tbl_name = next(iter(context.tables), None)
                        if tbl_name:
                            row = rows[orig_idx]
                            tbl = Table(source_cols, [row])
                            ctx = self.context({tbl_name: tbl})
                            ctx.set_index(0)
                            code = self.generate(default_expr)
                            default = ctx.eval(code) if code else None
                    target_pos = pos - offset if isinstance(func, exp.Lag) else pos + offset
                    if 0 <= target_pos < len(sorted_indices):
                        tbl_name = next(iter(context.tables), None)
                        if tbl_name:
                            row = rows[sorted_indices[target_pos]]
                            tbl = Table(source_cols, [row])
                            ctx = self.context({tbl_name: tbl})
                            ctx.set_index(0)
                            code = self.generate(func.this)
                            results[orig_idx] = ctx.eval(code) if code else None
                        else:
                            results[orig_idx] = default
                    else:
                        results[orig_idx] = default

                elif isinstance(func, exp.FirstValue):
                    vals = get_frame_values(pos, func.this, sorted_indices)
                    results[orig_idx] = next((v for v in vals), None)

                elif isinstance(func, exp.LastValue):
                    vals = get_frame_values(pos, func.this, sorted_indices)
                    results[orig_idx] = next((v for v in reversed(vals)), None)

                elif isinstance(func, exp.Count):
                    col_expr = func.this
                    if isinstance(col_expr, exp.Star):
                        lo, hi = frame_bounds(pos, sorted_indices)
                        results[orig_idx] = hi - lo + 1
                    else:
                        vals = get_frame_values(pos, col_expr, sorted_indices)
                        results[orig_idx] = sum(1 for v in vals if v is not None)

                elif isinstance(func, exp.Sum):
                    vals = get_frame_values(pos, func.this, sorted_indices)
                    non_null = [v for v in vals if v is not None]
                    results[orig_idx] = sum(non_null) if non_null else None

                elif isinstance(func, exp.Avg):
                    vals = get_frame_values(pos, func.this, sorted_indices)
                    non_null = [v for v in vals if v is not None]
                    results[orig_idx] = sum(non_null) / len(non_null) if non_null else None

                elif isinstance(func, exp.Min):
                    vals = get_frame_values(pos, func.this, sorted_indices)
                    non_null = [v for v in vals if v is not None]
                    results[orig_idx] = min(non_null) if non_null else None

                elif isinstance(func, exp.Max):
                    vals = get_frame_values(pos, func.this, sorted_indices)
                    non_null = [v for v in vals if v is not None]
                    results[orig_idx] = max(non_null) if non_null else None

        return results

    def set_operation(self, step, context):
        left = context.tables[step.left]
        right = context.tables[step.right]

        sink = self.table(left.columns)

        if issubclass(step.op, exp.Intersect):
            right_counts = collections.Counter(right.rows)
            seen = set()
            for row in left.rows:
                if right_counts[row] and (not step.distinct or row not in seen):
                    sink.append(row)
                    seen.add(row)
                    if not step.distinct:
                        right_counts[row] -= 1
        elif issubclass(step.op, exp.Except):
            right_counts = collections.Counter(right.rows)
            seen = set()
            for row in left.rows:
                if right_counts[row] and not step.distinct:
                    right_counts[row] -= 1
                elif not right_counts[row] and (not step.distinct or row not in seen):
                    sink.append(row)
                    seen.add(row)
        elif issubclass(step.op, exp.Union) and step.distinct:
            sink.rows = list(set(left.rows).union(set(right.rows)))
        else:
            sink.rows = left.rows + right.rows

        if not math.isinf(step.limit):
            sink.rows = sink.rows[0 : step.offset + step.limit]

        return self.context({step.name: sink})


class Python(Dialect):
    class Tokenizer(tokens.Tokenizer):
        STRING_ESCAPES = ["\\"]

    Generator = PythonGenerator
