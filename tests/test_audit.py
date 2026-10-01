import pandas as pd

from wonyotti_fr.audit import load_wallet


def test_wallet_cancellation_shared_snapshot_and_precision(tmp_path):
    frame = pd.DataFrame([
        ["2020-01-01", "XBt", "1000000", "Completed", "Deposit", "1000000"],
        ["2020-01-01", "XBt", "-500000", "Canceled", "Withdrawal", "1000000"],
        ["2020-01-02", "XBt", "300", "Completed", "RealisedPNL", "1.001E+6"],
        ["2020-01-02", "XBt", "400", "Completed", "RealisedPNL", "1.001E+6"],
        ["2020-01-03", "XBt", "-700", "Completed", "Withdrawal", "1000000"],
    ], columns=["date", "currency", "amount", "transactstatus", "transacttype", "walletbalance"])
    frame.to_csv(tmp_path / "aoa-wallet-test.csv", index=False)
    data, summary = load_wallet(tmp_path)
    assert data.posted_amount.sum() == 1000000
    assert summary["final_posted_balance_difference_satoshi"] == 0
    assert summary["rounded_balance_rows"] == 2
    assert summary["daily_snapshot_mismatch_beyond_precision"] == 0
