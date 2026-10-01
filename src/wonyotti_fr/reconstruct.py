from __future__ import annotations

from decimal import Decimal, getcontext

import pandas as pd

getcontext().prec = 36
D = Decimal


def reconstruct(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    quantity = 0
    basis = D(0)
    gross = D(0)
    fees = D(0)
    funding = D(0)
    episodes, actions, mismatches = [], [], []
    episode = None
    seen_orders: set[str] = set()
    add_orders: set[str] = set()
    episode_id = 0

    def start(time, direction, price, initial_qty):
        nonlocal episode_id, seen_orders, add_orders
        episode_id += 1
        seen_orders, add_orders = set(), set()
        return {
            "episode_id": episode_id, "entry_time": time, "exit_time": None,
            "direction": direction, "entry_price": price, "initial_qty": initial_qty,
            "max_qty": initial_qty, "gross": D(0), "fees": D(0), "funding": D(0),
            "fills": 0, "maker_fills": 0, "taker_fills": 0, "closed": False,
        }

    def finish(time, closed):
        assert episode is not None
        episode["exit_time"] = time
        episode["closed"] = closed
        episode["hold_minutes"] = (time - episode["entry_time"]).total_seconds() / 60
        episode["order_count"] = len(seen_orders)
        episode["additional_entry_orders"] = len(add_orders)
        episode["net_pnl_btc"] = float((episode["gross"] - episode["fees"] - episode["funding"]) / D(10**8))
        for key in ["gross", "fees", "funding"]:
            episode[f"{key}_btc"] = float(episode.pop(key) / D(10**8))
        episodes.append(episode.copy())

    for row in events.itertuples(index=False):
        before = quantity
        change = 0
        realized = D(0)
        if row.exectype == "Funding":
            paid = D(int(row.fee_satoshi))
            funding += paid
            if abs(quantity) != abs(int(row.quantity)):
                mismatches.append({"time": row.time, "position": quantity, "funding_qty": int(row.quantity)})
            if episode is not None:
                episode["funding"] += paid
            action = "funding"
        else:
            fill_qty = int(row.quantity)
            if fill_qty <= 0 or row.side not in {"Buy", "Sell"}:
                raise ValueError("잘못된 체결 방향 또는 수량")
            direction = 1 if row.side == "Buy" else -1
            change = direction * fill_qty
            unit_cost = abs(D(int(row.cost_satoshi))) / D(fill_qty)
            fee = D(int(row.fee_satoshi))
            fees += fee
            if quantity == 0 or (quantity > 0) == (change > 0):
                action = "open" if quantity == 0 else "increase"
                if episode is None:
                    episode = start(row.time, direction, float(row.price), fill_qty)
                elif row.order_key not in seen_orders:
                    add_orders.add(row.order_key)
                basis = (basis * abs(quantity) + unit_cost * fill_qty) / (abs(quantity) + fill_qty)
                quantity += change
                episode["fees"] += fee
            else:
                close_qty = min(abs(quantity), fill_qty)
                direction_before = 1 if quantity > 0 else -1
                realized = D(direction_before * close_qty) * (basis - unit_cost)
                gross += realized
                assert episode is not None
                episode["gross"] += realized
                episode["fees"] += fee * D(close_qty) / D(fill_qty)
                quantity += change
                action = "close" if quantity == 0 else "reduce"
                if fill_qty >= abs(before):
                    seen_orders.add(row.order_key)
                    episode["fills"] += int(row.fills)
                    episode["maker_fills"] += int(row.maker_fills)
                    episode["taker_fills"] += int(row.taker_fills)
                    finish(row.time, True)
                    episode = None
                    basis = D(0)
                    if quantity:
                        action = "reverse"
                        episode = start(row.time, direction, float(row.price), abs(quantity))
                        episode["fees"] += fee * D(abs(quantity)) / D(fill_qty)
                        basis = unit_cost
                    else:
                        actions.append({"time": row.time, "order_key": row.order_key,
                                        "action": action, "before_qty": before, "after_qty": quantity,
                                        "price": float(row.price), "realized_btc": float(realized / D(10**8)),
                                        "fee_btc": float(fee / D(10**8)), "episode_id": episode_id})
                        continue
            if episode is not None:
                seen_orders.add(row.order_key)
                episode["max_qty"] = max(episode["max_qty"], abs(quantity))
                episode["fills"] += int(row.fills)
                episode["maker_fills"] += int(row.maker_fills)
                episode["taker_fills"] += int(row.taker_fills)
        actions.append({"time": row.time, "order_key": row.order_key,
                        "action": action, "before_qty": before, "after_qty": quantity,
                        "price": float(row.price), "realized_btc": float(realized / D(10**8)),
                        "fee_btc": float(D(int(row.fee_satoshi)) / D(10**8)),
                        "episode_id": episode_id})
    if episode is not None:
        finish(events.iloc[-1].time, False)
    total_net = gross - fees - funding
    audit = {
        "initial_position_assumption": 0, "final_position_contracts": quantity,
        "final_basis_satoshi_per_contract": str(basis),
        "realized_gross_btc": float(gross / D(10**8)), "trade_fees_btc": float(fees / D(10**8)),
        "funding_cost_btc": float(funding / D(10**8)), "realized_net_btc": float(total_net / D(10**8)),
        "realized_net_satoshi_decimal": str(total_net),
        "funding_position_mismatch_count": len(mismatches), "funding_position_mismatches": mismatches[:30],
        "rounding": "원본 execcost와 execcomm의 사토시 단위로 계산. 부분 청산 원가는 Decimal 가중평균.",
        "episode_note": "동일 시각 이벤트는 원본 행 순서. 반전 체결 건수는 양쪽 에피소드에 중복될 수 있음.",
    }
    return pd.DataFrame(episodes), pd.DataFrame(actions), audit
