from src.verifiers.sql_verifier import SQLVerifier


DATABASE_SQL = """
CREATE TABLE employees (
    id INTEGER,
    name TEXT,
    department TEXT,
    salary REAL
);

INSERT INTO employees VALUES
    (1, 'Alice', 'IT', 60000),
    (2, 'Bob', 'IT', 70000),
    (3, 'Carol', 'HR', 50000),
    (4, 'David', 'HR', 55000);
"""


def test_correct_sql():
    prediction = """
    <think>Calculate the total salary for each department.</think>
    <answer>
    SELECT department, SUM(salary)
    FROM employees
    GROUP BY department
    </answer>
    """

    reference = {
        "database_sql": DATABASE_SQL,
        "reference_sql": """
            SELECT department, SUM(salary)
            FROM employees
            GROUP BY department
        """,
    }

    assert SQLVerifier.verify(prediction, reference) is True


def test_incorrect_sql():
    prediction = """
    <think>Calculate the average salary.</think>
    <answer>
    SELECT department, AVG(salary)
    FROM employees
    GROUP BY department
    </answer>
    """

    reference = {
        "database_sql": DATABASE_SQL,
        "reference_sql": """
            SELECT department, SUM(salary)
            FROM employees
            GROUP BY department
        """,
    }

    assert SQLVerifier.verify(prediction, reference) is False


def test_equivalent_sql_is_accepted():
    prediction = """
    <think>Return the IT employees.</think>
    <answer>
    SELECT name
    FROM employees
    WHERE department = 'IT'
    </answer>
    """

    reference = {
        "database_sql": DATABASE_SQL,
        "reference_sql": """
            SELECT employees.name
            FROM employees
            WHERE employees.department = 'IT'
        """,
    }

    assert SQLVerifier.verify(prediction, reference) is True


def test_missing_answer_is_rejected():
    prediction = """
    <think>The answer should contain an SQL query.</think>
    """

    reference = {
        "database_sql": DATABASE_SQL,
        "reference_sql": "SELECT COUNT(*) FROM employees",
    }

    assert SQLVerifier.verify(prediction, reference) is False


def test_invalid_sql_is_rejected():
    prediction = """
    <think>This query is invalid.</think>
    <answer>
    SELECT missing_column FROM employees
    </answer>
    """

    reference = {
        "database_sql": DATABASE_SQL,
        "reference_sql": "SELECT COUNT(*) FROM employees",
    }

    assert SQLVerifier.verify(prediction, reference) is False


def test_write_operation_is_rejected():
    prediction = """
    <think>Delete the HR employees.</think>
    <answer>
    DELETE FROM employees WHERE department = 'HR'
    </answer>
    """

    reference = {
        "database_sql": DATABASE_SQL,
        "reference_sql": "SELECT COUNT(*) FROM employees",
    }

    assert SQLVerifier.verify(prediction, reference) is False


def test_duplicate_rows_are_preserved():
    prediction = """
    <think>Return every department value.</think>
    <answer>
    SELECT department FROM employees
    </answer>
    """

    reference = {
        "database_sql": DATABASE_SQL,
        "reference_sql": """
            SELECT department FROM employees
        """,
    }

    assert SQLVerifier.verify(prediction, reference) is True


def test_invalid_reference_is_rejected():
    prediction = """
    <think>Return the number of employees.</think>
    <answer>
    SELECT COUNT(*) FROM employees
    </answer>
    """

    assert SQLVerifier.verify(
        prediction,
        {},
    ) is False