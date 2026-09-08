from src.trainer.callbacks import RewardHackingMonitor


def test_reward_hacking_monitor_records_current_length_metric():
    monitor = RewardHackingMonitor(window_size=2)

    logs = {
        "reward": 0.5,
        "completions/mean_length": 100,
    }

    monitor.on_log(None, None, None, logs=logs)

    assert len(monitor.history) == 1
    assert monitor.history[0] == (100.0, 0.5)


def test_reward_hacking_monitor_ignores_missing_length_metric():
    monitor = RewardHackingMonitor(window_size=2)

    logs = {
        "reward": 0.5,
        "completion_length": 100,
    }

    monitor.on_log(None, None, None, logs=logs)

    assert len(monitor.history) == 0


def test_reward_hacking_monitor_handles_zero_variance():
    monitor = RewardHackingMonitor(window_size=2)

    logs = {
        "reward": 0.5,
        "completions/mean_length": 100,
    }

    monitor.on_log(None, None, None, logs=logs)
    monitor.on_log(None, None, None, logs=logs)

    assert len(monitor.history) == 2