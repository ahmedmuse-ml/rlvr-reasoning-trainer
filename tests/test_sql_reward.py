from src.rewards.accuracy_reward import AccuracyReward


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


REFERENCE_SQL = """
SELECT department, SUM(salary)
FROM employees
GROUP BY department
"""


def test_sql_accuracy_reward_correct():
    reward = AccuracyReward()

    completion = """
    <think>Calculate total salary by department.</think>
    <answer>
    SELECT department, SUM(salary)
    FROM employees
    GROUP BY department
    </answer>
    """

    rewards = reward.compute(
        [completion],
        context={
            "answer": [""],
            "test_list": [[]],
            "domain": ["sql"],
            "database_sql": [DATABASE_SQL],
            "reference_sql": [REFERENCE_SQL],
        },
    )

    assert rewards == [1.0]


def test_sql_accuracy_reward_incorrect():
    reward = AccuracyReward()

    completion = """
    <think>Calculate average salary by department.</think>
    <answer>
    SELECT department, AVG(salary)
    FROM employees
    GROUP BY department
    </answer>
    """

    rewards = reward.compute(
        [completion],
        context={
            "answer": [""],
            "test_list": [[]],
            "domain": ["sql"],
            "database_sql": [DATABASE_SQL],
            "reference_sql": [REFERENCE_SQL],
        },
    )

    assert rewards == [0.0]


def test_sql_accuracy_reward_invalid_query():
    reward = AccuracyReward()

    completion = """
    <think>Use a nonexistent column.</think>
    <answer>
    SELECT nonexistent FROM employees
    </answer>
    """

    rewards = reward.compute(
        [completion],
        context={
            "answer": [""],
            "test_list": [[]],
            "domain": ["sql"],
            "database_sql": [DATABASE_SQL],
            "reference_sql": [REFERENCE_SQL],
        },
    )

    assert rewards == [0.0]


def test_sql_accuracy_reward_mixed_domains():
    reward = AccuracyReward()

    completions = [
        "<think>5+5=10</think><answer>10</answer>",
        """
        <think>Calculate total salary by department.</think>
        <answer>
        SELECT department, SUM(salary)
        FROM employees
        GROUP BY department
        </answer>
        """,
    ]

    rewards = reward.compute(
        completions,
        context={
            "answer": ["10", ""],
            "test_list": [[], []],
            "domain": ["math", "sql"],
            "database_sql": ["", DATABASE_SQL],
            "reference_sql": ["", REFERENCE_SQL],
        },
    )

    assert rewards == [1.0, 1.0]