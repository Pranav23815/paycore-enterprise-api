"""
FILE 2: app.py
Enterprise Payroll REST API — Flask + psycopg2 (no ORM)

Architecture notes
──────────────────
• get_db_conn()   — thin connection factory; keeps connection logic in one place.
• Each route uses a context-manager cursor so connections are always returned
  to the pool cleanly, even on exceptions.
• /api/employees/new uses an explicit transaction block (try/except with
  ROLLBACK) to guarantee atomic dual-insert (employee + payroll row).
• Parameterised queries throughout — zero string interpolation into SQL.
"""

import os
import json
import logging
from contextlib import contextmanager
from datetime import date

import psycopg2
import psycopg2.extras          # RealDictCursor
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── App setup ───────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow all origins for local dev; lock this down in production.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Database configuration ──────────────────────────────────────────────────
# Override any of these with real environment variables in prod/CI.

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "payroll_db"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "pranav123"),
}

# ─── Connection helpers ───────────────────────────────────────────────────────

def get_db_conn():
    """Open a new psycopg2 connection using DB_CONFIG."""
    return psycopg2.connect(**DB_CONFIG)


@contextmanager
def db_cursor(conn=None, commit=False):
    """
    Context manager that yields a RealDictCursor and handles cleanup.

    Usage (read-only):
        with db_cursor() as cur:
            cur.execute(...)

    Usage (write, auto-commit on success):
        with db_cursor(commit=True) as cur:
            cur.execute(...)
    """
    _conn = conn or get_db_conn()
    try:
        with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
            if commit:
                _conn.commit()
    except Exception:
        _conn.rollback()
        raise
    finally:
        if conn is None:          # We opened it, we close it.
            _conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — GET /api/departments/cost
# Returns total payroll cost (base_salary + bonus) per department.
# Technique: JOIN across three tables + GROUP BY aggregation.
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/departments/cost", methods=["GET"])
def department_cost():
    """
    SQL pattern: JOIN + GROUP BY aggregate.

    Returns each department alongside:
      • headcount   — number of employees with a payroll record
      • total_cost  — SUM(base_salary + bonus)
      • avg_salary  — AVG(base_salary) for context
    """
    sql = """
        SELECT
            d.dept_id,
            d.name                                        AS department,
            COUNT(DISTINCT e.emp_id)                      AS headcount,
            SUM(p.base_salary + p.bonus)                  AS total_cost,
            ROUND(AVG(p.base_salary), 2)                  AS avg_base_salary
        FROM  departments  d
        JOIN  employees    e  ON e.dept_id = d.dept_id
        JOIN  payroll      p  ON p.emp_id  = e.emp_id
        GROUP BY d.dept_id, d.name
        ORDER BY total_cost DESC;
    """
    try:
        with db_cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        # psycopg2 returns Decimal for NUMERIC columns — cast for JSON safety.
        result = [
            {
                "dept_id":        r["dept_id"],
                "department":     r["department"],
                "headcount":      r["headcount"],
                "total_cost":     float(r["total_cost"]),
                "avg_base_salary": float(r["avg_base_salary"]),
            }
            for r in rows
        ]
        return jsonify({"status": "ok", "data": result}), 200

    except psycopg2.Error as e:
        logger.error("DB error in /api/departments/cost: %s", e)
        return jsonify({"status": "error", "message": "Database error"}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — GET /api/employees/top-earners
# Returns the #1 highest-paid employee per department.
# Technique: Window Function — ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY ...)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/employees/top-earners", methods=["GET"])
def top_earners():
    """
    SQL pattern: CTE + ROW_NUMBER() window function.

    Step 1 (CTE ranked): Assign every employee a rank within their department,
            ordered by (base_salary + bonus) descending.
    Step 2 (outer query): Filter WHERE rn = 1 to keep only the top earner
            per department.

    This pattern is far more robust than a correlated sub-query and scales
    cleanly to top-N with a single WHERE rn <= N change.
    """
    sql = """
        WITH ranked AS (
            SELECT
                e.emp_id,
                e.name                                        AS employee_name,
                e.role,
                d.name                                        AS department,
                d.dept_id,
                p.base_salary,
                p.bonus,
                (p.base_salary + p.bonus)                     AS total_compensation,
                ROW_NUMBER() OVER (
                    PARTITION BY e.dept_id
                    ORDER BY (p.base_salary + p.bonus) DESC
                )                                             AS rn
            FROM  employees e
            JOIN  departments d  ON d.dept_id = e.dept_id
            JOIN  payroll     p  ON p.emp_id  = e.emp_id
        )
        SELECT
            emp_id,
            employee_name,
            role,
            department,
            dept_id,
            base_salary,
            bonus,
            total_compensation
        FROM  ranked
        WHERE rn = 1
        ORDER BY total_compensation DESC;
    """
    try:
        with db_cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        result = [
            {
                "emp_id":              r["emp_id"],
                "employee_name":       r["employee_name"],
                "role":                r["role"],
                "department":          r["department"],
                "dept_id":             r["dept_id"],
                "base_salary":         float(r["base_salary"]),
                "bonus":               float(r["bonus"]),
                "total_compensation":  float(r["total_compensation"]),
            }
            for r in rows
        ]
        return jsonify({"status": "ok", "data": result}), 200

    except psycopg2.Error as e:
        logger.error("DB error in /api/employees/top-earners: %s", e)
        return jsonify({"status": "error", "message": "Database error"}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — POST /api/employees/new
# Atomically inserts a new employee + their payroll record.
# Technique: Explicit transaction with ROLLBACK on any failure.
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/employees/new", methods=["POST"])
def create_employee():
    """
    Expects JSON body:
    {
        "name":        "Jane Doe",
        "role":        "Staff Engineer",
        "dept_id":     1,
        "base_salary": 150000,
        "bonus":       12000
    }

    Transaction guarantee
    ─────────────────────
    Both the employees INSERT and the payroll INSERT must succeed or
    neither is committed. If payroll INSERT fails (e.g. FK violation,
    constraint) we ROLLBACK and the employee row is also undone — the
    database is never left in a half-written state.
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"status": "error", "message": "JSON body required"}), 400

    # ── Input validation ──────────────────────────────────────────────────────
    required = ("name", "role", "dept_id", "base_salary", "bonus")
    missing  = [f for f in required if f not in body]
    if missing:
        return jsonify({
            "status":  "error",
            "message": f"Missing fields: {', '.join(missing)}"
        }), 422

    try:
        name        = str(body["name"]).strip()
        role        = str(body["role"]).strip()
        dept_id     = int(body["dept_id"])
        base_salary = float(body["base_salary"])
        bonus       = float(body["bonus"])
    except (ValueError, TypeError) as e:
        return jsonify({"status": "error", "message": f"Invalid field types: {e}"}), 422

    if not name or not role:
        return jsonify({"status": "error", "message": "name and role cannot be blank"}), 422
    if base_salary < 0 or bonus < 0:
        return jsonify({"status": "error", "message": "Salary/bonus must be >= 0"}), 422

    # ── Atomic transaction ────────────────────────────────────────────────────
    conn = get_db_conn()
    try:
        conn.autocommit = False          # Begin implicit transaction

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # Step 1: Insert employee, return the generated emp_id via RETURNING.
            cur.execute(
                """
                INSERT INTO employees (name, role, dept_id)
                VALUES (%s, %s, %s)
                RETURNING emp_id;
                """,
                (name, role, dept_id),
            )
            new_emp_id = cur.fetchone()["emp_id"]

            # Step 2: Insert the initial payroll record for this employee.
            cur.execute(
                """
                INSERT INTO payroll (emp_id, base_salary, bonus, payment_date)
                VALUES (%s, %s, %s, %s)
                RETURNING transaction_id;
                """,
                (new_emp_id, base_salary, bonus, date.today()),
            )
            new_txn_id = cur.fetchone()["transaction_id"]

        conn.commit()          # ← Both inserts land atomically here.
        logger.info(
            "Created employee emp_id=%s, transaction_id=%s", new_emp_id, new_txn_id
        )

        return jsonify({
            "status":         "ok",
            "message":        "Employee and payroll record created successfully.",
            "emp_id":         new_emp_id,
            "transaction_id": new_txn_id,
        }), 201

    except psycopg2.errors.ForeignKeyViolation:
        conn.rollback()
        logger.warning("FK violation: dept_id=%s does not exist", dept_id)
        return jsonify({
            "status":  "error",
            "message": f"dept_id {dept_id} does not exist."
        }), 409

    except psycopg2.Error as e:
        conn.rollback()        # ← Both inserts are undone on any DB error.
        logger.error("DB error in /api/employees/new: %s", e)
        return jsonify({"status": "error", "message": "Database error — transaction rolled back."}), 500

    finally:
        conn.close()


# ─── Health check ─────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    try:
        with db_cursor() as cur:
            cur.execute("SELECT 1;")
        return jsonify({"status": "ok", "db": "reachable"}), 200
    except psycopg2.Error as e:
        return jsonify({"status": "error", "db": str(e)}), 503


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)