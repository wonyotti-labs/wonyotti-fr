from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import pandas as pd

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_diagnostics import decompose_run
from .event_research import load_selection
from .minute_data import prepare_minute_period
from .pullback_diagnostics import waiting_diagnostics
from .pullback_policy import PullbackPolicy
from .reports import table
from .robustness import block_interval

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def evaluation_period(frozen: dict, period: str) -> tuple[str, str]:
    allowed = {'observed': ('2022-01-01', '2026-01-01'), 'seen_2026': ('2026-01-01', '2026-10-01'),
               'verified_2022_2024': ('2022-01-01', '2025-01-01')}
    if frozen.get('protocol') in {'net_edge_v8', 'lifecycle_v9', 'minute_action_v10', 'minute_action_v11', 'minute_path_v12', 'minute_reverse_v13', 'minute_rate_v14', 'minute_rate_reverse_v18', 'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46', 'lifecycle_edge_v15', 'lifecycle_edge_v16', 'lifecycle_edge_v17'}:
        allowed = {'observed': ('2023-01-01', '2026-01-01'), 'seen_2026': ('2026-01-01', '2026-10-01')}
    if frozen.get('protocol') not in {'pullback_v7', 'net_edge_v8', 'lifecycle_v9', 'minute_action_v10', 'minute_action_v11', 'minute_path_v12', 'minute_reverse_v13', 'minute_rate_v14', 'minute_rate_reverse_v18', 'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46', 'lifecycle_edge_v15', 'lifecycle_edge_v16', 'lifecycle_edge_v17'} or period not in allowed:
        raise ValueError('v7~v36 고정 후보와 이미 관찰한 평가 기간이 필요합니다.')
    key = 'seen_2026_period' if period == 'seen_2026' else 'observed_evaluation_period'
    expected = allowed['observed'] if period == 'verified_2022_2024' else allowed[period]
    if (tuple(frozen[key]) != expected or frozen['evaluation_end_exclusive'] != '2026-10-01'
        or frozen['unseen_evaluation_available'] is not False or frozen['risk']['bar_seconds'] != 60):
        raise ValueError('v7 고정 후보의 평가 범위·실행 간격 오류')
    return allowed[period]


def run_pullback_evaluation(selection: Path, market: Path, feature_market: Path, output: Path,
                            period: str, symbols: list[str], diagnostic_only: bool = False) -> Path:
    frozen, policy = load_selection(selection)
    is_edge = frozen['protocol'] in {'lifecycle_edge_v15', 'lifecycle_edge_v16', 'lifecycle_edge_v17'}
    is_net = frozen['protocol'] == 'net_edge_v8'
    is_action = frozen['protocol'] in {'minute_action_v10', 'minute_action_v11', 'minute_path_v12', 'minute_reverse_v13', 'minute_rate_v14', 'minute_rate_reverse_v18', 'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46', 'lifecycle_edge_v15', 'lifecycle_edge_v16', 'lifecycle_edge_v17'}
    is_lifecycle = frozen['protocol'] in {'lifecycle_v9', 'minute_action_v10', 'minute_action_v11', 'minute_path_v12', 'minute_reverse_v13', 'minute_rate_v14', 'minute_rate_reverse_v18', 'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46', 'lifecycle_edge_v15', 'lifecycle_edge_v16', 'lifecycle_edge_v17'}
    start, end = evaluation_period(frozen, period)
    if not symbols or len(symbols) != len(set(symbols)) or not set(symbols) <= {'BTCUSDT', 'ETHUSDT', 'SOLUSDT'}:
        raise ValueError('평가 심볼의 종류·중복 오류')
    config = EngineConfig(**frozen['risk'])
    variants = {
        'fixed_policy': (policy, config),
        ('ungated_v7' if is_net else 'immediate'): (PullbackPolicy(policy.base, policy.offset_bps, policy.ttl_minutes, None if is_net else 'immediate'), config),
        'cash': (PullbackPolicy(policy.base, policy.offset_bps, policy.ttl_minutes, 'cash'), config),
        'cost_x2': (policy, replace(config, fee_bps=10, slippage_bps=6)),
        'cost_x3': (policy, replace(config, fee_bps=15, slippage_bps=9)),
        'extra_minute_delay': (policy, replace(config, signal_delay_bars=1)),
    }
    if is_lifecycle:
        previous = json.loads((selection / 'pullback_selection.json').read_text())
        variants = {'fixed_policy': (policy, config),
                    'ungated_v7': (PullbackPolicy(policy.base, policy.offset_bps, policy.ttl_minutes), EngineConfig(**previous['risk'])),
                    'cash': variants['cash'], 'no_adds': (policy, replace(config, max_adds=0)),
                    'cap_30m': (policy, replace(config, max_hold_bars=30)),
                    **{key: variants[key] for key in ['cost_x2', 'cost_x3', 'extra_minute_delay']}}
    if frozen['protocol'] == 'minute_rate_reverse_v18':
        from .action_research import load_action_selection
        _, original_manager = load_action_selection(selection, json.loads((selection / 'rate_selection.json').read_text()))
        variants['unfiltered_v14'] = (original_manager, config)
    if frozen['protocol'] in {'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .action_research import load_action_selection
        from .inventory_research import fixed_size_control
        _, original_manager = load_action_selection(selection, json.loads((selection / 'rate_selection.json').read_text()))
        variants['unfiltered_v14'] = (original_manager, config)
        if frozen['protocol'] not in {'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
            variants['inventory_fixed_size'] = (fixed_size_control(policy), config)
    if frozen['protocol'] in {'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        _, previous_inventory = load_selection(selection / 'previous_inventory')
        variants['previous_v20'] = (previous_inventory, config)
    if frozen['protocol'] == 'addition_effect_v22':
        from .minute_inventory_research import load_minute_inventory_selection
        _, original_minute = load_minute_inventory_selection(selection, json.loads((selection / 'minute_selection.json').read_text()))
        variants['unfiltered_v21'] = (original_minute, config)
    if frozen['protocol'] in {'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .minute_inventory_research import load_minute_inventory_selection
        _, original_minute = load_minute_inventory_selection(selection, json.loads((selection / 'minute_selection.json').read_text()))
        variants['previous_v21'] = (original_minute, config)
        variants['ungated_v7'] = (PullbackPolicy(original_minute.base, policy.offset_bps, policy.ttl_minutes), EngineConfig(**previous['risk']))
    if frozen['protocol'] in {'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .new_position import load_new_position_selection
        _, original_position = load_new_position_selection(selection, json.loads((selection / 'position_selection.json').read_text()))
        variants['previous_v25'] = (original_position, config)
        variants.pop('previous_v20')
    if frozen['protocol'] in {'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .position_prior import load_position_prior_selection
        _, original_prior = load_position_prior_selection(selection, json.loads((selection / 'prior_selection.json').read_text()))
        variants['previous_v26'] = (original_prior, config)
        variants.pop('previous_v21')
    if frozen['protocol'] in {'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .position_direction import load_position_direction_selection
        _, original_direction = load_position_direction_selection(selection, json.loads((selection / 'direction_selection.json').read_text()))
        variants['previous_v27'] = (original_direction, config)
        variants.pop('previous_v25')
    if frozen['protocol'] in {'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .probe_entry import load_probe_parent, probe_risks
        parent, original_probe = load_probe_parent(selection, frozen)
        _, original_risk, matched_risk = probe_risks(parent)
        variants['previous_v28'] = (original_probe, original_risk)
        variants['matched_initial_risk'] = (original_probe, matched_risk)
        variants['unfiltered_v14'] = (variants['unfiltered_v14'][0], original_risk)
        variants.pop('previous_v26')
        variants.pop('previous_v27')
    if frozen['protocol'] in {'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .current_state import load_current_parent
        _, original_current = load_current_parent(selection, frozen)
        variants['previous_v29'] = (original_current, config)
        variants.pop('matched_initial_risk')
    if frozen['protocol'] in {'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .history_state import load_history_parent
        _, previous_history = load_history_parent(selection, frozen)
        variants['previous_v31'] = (previous_history, config)
        variants.pop('previous_v28')
    if frozen['protocol'] in {'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .first_state import load_first_parent
        _, original_first = load_first_parent(selection, frozen)
        variants['previous_v36'] = (original_first, config)
        variants.pop('previous_v29')
    if frozen['protocol'] in {'holding_support_v40', 'activity_ablation_v46'}:
        from .holding_support import load_holding_parent
        first_parent, original_holding = load_holding_parent(selection, frozen)
        parent_risk = EngineConfig(**first_parent['risk'])
        variants['previous_v39'] = (original_holding, parent_risk)
        variants['previous_v36'] = (variants['previous_v36'][0], parent_risk)
        variants.pop('previous_v31')
    if frozen['protocol'] == 'activity_ablation_v46':
        from .activity_ablation import load_activity_parent
        activity_parent, original_activity = load_activity_parent(selection, frozen)
        variants['previous_v40'] = (original_activity, EngineConfig(**activity_parent['risk']))
        variants.pop('previous_v36')
    if frozen['protocol'] == 'new_position_v25':
        from .recent_entry import load_recent_entry_selection
        _, original_recent = load_recent_entry_selection(selection, json.loads((selection / 'recent_selection.json').read_text()))
        variants['previous_v23'] = (original_recent, config)
        variants.pop('previous_v20')
    if is_edge:
        variants['unfiltered_v14'] = (policy.manager, config)
    if frozen['protocol'] == 'lifecycle_edge_v17':
        from .action_research import load_action_selection
        from .pullback_research import load_pullback_selection
        _, original_entry = load_pullback_selection(selection, previous)
        _, original_manager = load_action_selection(selection, json.loads((selection / 'rate_selection.json').read_text()))
        variants['ungated_v7'] = (original_entry, EngineConfig(**previous['risk']))
        variants['unfiltered_v14'] = (original_manager, config)
        variants['unfiltered_direction_only'] = (policy.manager, config)
    protocol = 'docs/EXPERIMENT_V9.md' if is_lifecycle else ('docs/EXPERIMENT_V8.md' if is_net else 'docs/EXPERIMENT_V7.md')
    label = 'lifecycle' if is_lifecycle else ('net-edge' if is_net else 'pullback')
    if is_action:
        protocol, label = ('docs/EXPERIMENT_V11.md', 'action-recent') if frozen['protocol'] == 'minute_action_v11' else ('docs/EXPERIMENT_V10.md', 'action')
        if frozen['protocol'] == 'minute_path_v12':
            protocol, label = 'docs/EXPERIMENT_V12.md', 'action-path'
        if frozen['protocol'] == 'minute_reverse_v13':
            protocol, label = 'docs/EXPERIMENT_V13.md', 'action-reversal'
        if frozen['protocol'] == 'minute_rate_v14':
            protocol, label = 'docs/EXPERIMENT_V14.md', 'action-rate'
        if frozen['protocol'] == 'minute_rate_reverse_v18':
            protocol, label = 'docs/EXPERIMENT_V18.md', 'action-rate-reversal'
        if frozen['protocol'] in {'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
            protocol, label = (('docs/EXPERIMENT_V20.md', 'inventory-recent')
                               if frozen['protocol'] == 'minute_inventory_recent_v20' else ('docs/EXPERIMENT_V19.md', 'inventory'))
        if frozen['protocol'] in {'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
            protocol, label = 'docs/EXPERIMENT_V21.md', 'minute-inventory'
        if frozen['protocol'] == 'addition_effect_v22':
            protocol, label = 'docs/EXPERIMENT_V22.md', 'addition-effect'
        if frozen['protocol'] == 'recent_entry_v23':
            protocol, label = 'docs/EXPERIMENT_V23.md', 'recent-entry'
        if frozen['protocol'] == 'realized_exit_v24':
            protocol, label = 'docs/EXPERIMENT_V24.md', 'realized-exit'
        if frozen['protocol'] == 'new_position_v25':
            protocol, label = 'docs/EXPERIMENT_V25.md', 'new-position'
        if frozen['protocol'] == 'position_prior_v26':
            protocol, label = 'docs/EXPERIMENT_V26.md', 'position-prior'
        if frozen['protocol'] in {'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
            protocol, label = 'docs/EXPERIMENT_V27.md', 'position-direction'
        if frozen['protocol'] in {'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
            protocol, label = 'docs/EXPERIMENT_V28.md', 'boosted-direction'
        if frozen['protocol'] in {'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
            protocol, label = 'docs/EXPERIMENT_V29.md', 'probe-entry'
        if frozen['protocol'] in {'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
            protocol, label = 'docs/EXPERIMENT_V31.md', 'current-state'
        if frozen['protocol'] in {'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
            protocol, label = 'docs/EXPERIMENT_V36.md', 'history-state'
        if frozen['protocol'] in {'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
            protocol, label = 'docs/EXPERIMENT_V39.md', 'first-state'
        if frozen['protocol'] in {'holding_support_v40', 'activity_ablation_v46'}:
            protocol, label = 'docs/EXPERIMENT_V40.md', 'holding-support'
        if frozen['protocol'] == 'activity_ablation_v46':
            protocol, label = 'docs/EXPERIMENT_V46.md', 'activity-ablation'
        if is_edge:
            protocol, label = (('docs/EXPERIMENT_V16.md', 'lifecycle-weighted') if frozen['protocol'] == 'lifecycle_edge_v16'
                               else ('docs/EXPERIMENT_V15.md', 'lifecycle-edge'))
            if frozen['protocol'] == 'lifecycle_edge_v17':
                protocol, label = 'docs/EXPERIMENT_V17.md', 'direction-edge'
    if diagnostic_only:
        if frozen['protocol'] not in {'minute_path_v12', 'minute_reverse_v13', 'minute_rate_v14', 'minute_rate_reverse_v18', 'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46', 'lifecycle_edge_v15', 'lifecycle_edge_v16', 'lifecycle_edge_v17'} or symbols != ['BTCUSDT']:
            raise ValueError('축소 진단은 지원하는 BTC 고정 비교만 허용합니다.')
        confirmation = json.loads((selection / 'confirmation-2022' / 'metrics.json').read_text())
        if confirmation['total_return'] > 0 and confirmation['closed_trades'] >= 30 and not confirmation['permanent_halt']:
            raise ValueError('확인 선행 조건을 통과한 후보는 전체 평가가 필요합니다.')
        variants = {'fixed_policy': variants['fixed_policy']}
        label += '-diagnostic'
    destination = new_run(output, f'{label}-evaluation-{period}', {
        'protocol': protocol, 'protocol_sha256': sha256(Path(protocol)), 'selection_sha256': sha256(selection / 'frozen_selection.json'),
        'market_manifest_sha256': sha256(market / 'manifest-1m.json'),
        'feature_manifest_sha256': sha256(feature_market / 'manifest-5m.json'),
        'period': [start, end], 'symbols': symbols, 'all_periods_already_observed': True, 'retuning': False,
        'diagnostic_only': diagnostic_only, 'all_variants_requested': not diagnostic_only,
        'variants': {name: risk.__dict__ for name, (_, risk) in variants.items()}, 'bootstrap': '30-day circular blocks, 1000, seed 41'})
    names = ['frozen_selection.json', 'frozen_integrity.json', 'base_selection.json', 'expansion_models.json']
    if is_net:
        names.extend(['pullback_selection.json', 'net_model.json'])
    if is_lifecycle:
        names.extend(['pullback_selection.json', 'action_model.json' if is_action else 'management_model.json'])
    if frozen['protocol'] in {'minute_reverse_v13', 'minute_rate_v14', 'minute_rate_reverse_v18', 'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46', 'lifecycle_edge_v15', 'lifecycle_edge_v16', 'lifecycle_edge_v17'}:
        names.append('path_selection.json')
    if frozen['protocol'] in {'minute_rate_v14', 'minute_rate_reverse_v18', 'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46', 'lifecycle_edge_v15', 'lifecycle_edge_v16', 'lifecycle_edge_v17'}:
        names.append('rate_calibration.json')
    if frozen['protocol'] == 'minute_rate_reverse_v18':
        names.append('rate_selection.json')
    if frozen['protocol'] in {'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .inventory_research import INVENTORY_FILES
        names.extend(['rate_selection.json', *INVENTORY_FILES])
    if is_edge:
        names.extend(['rate_selection.json', 'net_model.json'])
    if frozen['protocol'] in {'lifecycle_edge_v16', 'lifecycle_edge_v17'}:
        names.extend(['training_weights.parquet', 'weighting.json'])
    if frozen['protocol'] in {'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .minute_inventory_research import copy_previous_inventory
        copy_previous_inventory(selection / 'previous_inventory', destination / 'previous_inventory')
    if frozen['protocol'] == 'addition_effect_v22':
        from .addition_research import ADDITION_FILES
        names.extend(['minute_selection.json', *ADDITION_FILES])
    if frozen['protocol'] == 'recent_entry_v23':
        from .recent_entry import ENTRY_FILES
        names.extend(['minute_selection.json', *ENTRY_FILES])
    if frozen['protocol'] == 'realized_exit_v24':
        from .realized_exit_research import EXIT_MODEL_FILES
        names.extend(['minute_selection.json', *EXIT_MODEL_FILES])
    if frozen['protocol'] in {'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .new_position import POSITION_FILES
        from .recent_entry import ENTRY_FILES
        names.extend(['minute_selection.json', 'recent_selection.json', *ENTRY_FILES, *POSITION_FILES])
    if frozen['protocol'] in {'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .position_prior import PRIOR_FILES
        names.extend(['position_selection.json', *PRIOR_FILES])
    if frozen['protocol'] in {'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .position_direction import DIRECTION_FILES
        names.extend(['prior_selection.json', *DIRECTION_FILES])
    if frozen['protocol'] in {'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .boosted_direction import BOOST_FILES
        names.extend(['direction_selection.json', *BOOST_FILES])
    if frozen['protocol'] in {'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        names.append('boost_selection.json')
    if frozen['protocol'] in {'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        names.append('probe_selection.json')
    if frozen['protocol'] in {'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .history_state import HISTORY_POLICY_FILES
        names.extend(['current_selection.json', *HISTORY_POLICY_FILES])
    if frozen['protocol'] in {'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'}:
        from .first_state import FIRST_POLICY_FILES
        names.extend(['history_selection.json', *FIRST_POLICY_FILES])
    if frozen['protocol'] in {'holding_support_v40', 'activity_ablation_v46'}:
        names.extend(['first_selection.json', 'holding_support.json'])
    if frozen['protocol'] == 'activity_ablation_v46':
        names.append('holding_selection.json')
    for name in names:
        (destination / name).write_bytes((selection / name).read_bytes())
    print(f'진입 대기 {period} 평가: {destination}', flush=True)
    rows, annual, intervals, decompositions, waiting = [], [], [], [], []
    chart, axes = plt.subplots(len(symbols), 1, figsize=(12, 3.5 * len(symbols)), squeeze=False, layout='constrained')
    try:
        for axis, symbol in zip(axes[:, 0], symbols, strict=True):
            bars, checks = prepare_minute_period(market, feature_market, symbol, start, end,
                **({'minute_inputs': True} if frozen['protocol'] in {'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'current_state_v31', 'history_state_v36', 'first_state_v39', 'holding_support_v40', 'activity_ablation_v46'} else {}))
            save_json(destination / f'{symbol}-input.json', checks)
            for name, (strategy, risk) in variants.items():
                target = destination / symbol / name
                metrics = backtest(bars, strategy, risk, target)
                rows.append({'symbol': symbol, 'strategy': name, **metrics})
                save_json(destination / 'results_partial.json', rows)
                print(f'{symbol}/{name}: 수익 {metrics["total_return"]:.2%}, 낙폭 {metrics["max_drawdown"]:.2%}, 거래 {metrics["closed_trades"]}', flush=True)
                if is_lifecycle:
                    from .action_research import action_diagnostics
                    from .lifecycle_research import lifecycle_diagnostics
                    diagnose = action_diagnostics if is_action else lifecycle_diagnostics
                    if frozen['protocol'] == 'addition_effect_v22':
                        from .addition_research import addition_diagnostics
                        details = addition_diagnostics(target, bars, strategy, risk)
                    elif is_edge:
                        from .lifecycle_edge_research import lifecycle_edge_diagnostics
                        details = lifecycle_edge_diagnostics(target, bars, strategy, risk)
                    else:
                        details = diagnose(target, bars, risk)
                    decomposition, wait = details['decomposition'], details['waiting']
                elif is_net:
                    from .net_edge_research import net_diagnostics
                    details = net_diagnostics(target, bars, strategy, risk)
                    decomposition, wait = details['decomposition'], details['waiting']
                else:
                    decomposition = decompose_run(target, risk.initial_equity)
                    wait = waiting_diagnostics(target, bars, risk.signal_delay_bars)
                decompositions.append({'symbol': symbol, 'strategy': name, **decomposition})
                waiting.append({'symbol': symbol, 'strategy': name, **wait})
                curve = pd.read_parquet(target / 'equity.parquet', columns=['time', 'equity'])
                days = (pd.to_datetime(curve.time, utc=True) - pd.Timedelta(nanoseconds=1)).dt.floor('1D')
                daily = curve.groupby(days).equity.last()
                returns = (daily / daily.shift(1, fill_value=risk.initial_equity) - 1).to_numpy()
                intervals.append({'symbol': symbol, 'strategy': name, **block_interval(returns)})
                if name in {'fixed_policy', 'immediate', 'ungated_v7', 'unfiltered_v14', 'cash'}:
                    axis.plot(daily.index, daily / risk.initial_equity, label=name, lw=.9)
                save_json(destination / 'decomposition.json', decompositions)
                save_json(destination / 'waiting.json', waiting)
                save_json(destination / 'bootstrap.json', intervals)
            if period in {'observed', 'verified_2022_2024'}:
                for year in range(int(start[:4]), int(end[:4])):
                    part = bars[(bars.time >= f'{year}-01-01') & (bars.time < f'{year+1}-01-01')]
                    target = destination / symbol / f'restart-{year}'
                    metrics = backtest(part, policy, config, target)
                    decomposition = decompose_run(target, config.initial_equity)
                    waiting_diagnostics(target, part, config.signal_delay_bars, management_state=is_action)
                    if is_net:
                        net_diagnostics(target, part, policy, config)
                    if frozen['protocol'] == 'addition_effect_v22':
                        addition_diagnostics(target, part, policy, config)
                    elif is_edge:
                        lifecycle_edge_diagnostics(target, part, policy, config)
                    elif is_lifecycle:
                        diagnose(target, part, config)
                    annual.append({'symbol': symbol, 'year': year, **metrics, 'accounting_reconciled': True,
                                   'net_pnl': decomposition['net_pnl']})
                    save_json(destination / 'annual_restart.json', annual)
                    print(f'{symbol}/{year} 독립 재시작: {metrics["total_return"]:.2%}', flush=True)
            axis.set(title=f'{symbol}: fixed pullback policy ({period})', ylabel='Equity / initial')
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
        chart.savefig(destination / 'equity_comparison.png', dpi=150)
        save_json(destination / 'results.json', rows)
        pd.DataFrame(rows).drop(columns='rejected').to_csv(destination / 'results.csv', index=False)
        decision = {'profitability_review_candidate': False, 'live_trading_approved': False,
                    'reason': '이미 관찰한 기간의 고정 후보 비교이며 선행 조건 및 선택 불확실성 보존'}
        save_json(destination / 'decision.json', decision)
        summary = pd.DataFrame(rows)[['symbol', 'strategy', 'total_return', 'max_drawdown', 'closed_trades', 'fees', 'permanent_halt']]
        scope_note = ('2025년 입력 검증 실패 이후 결과 개봉 전에 고정한 2022~2024년 추가 비교다. '
                      '2022~2025년 전체 연속 평가 및 2025년 연간 평가는 완료하지 못했다.\n\n'
                      if period == 'verified_2022_2024' else '')
        if diagnostic_only:
            scope_note = '확인 선행 조건 실패 후 사전 계획한 BTC 고정·연간 진단이다. 다른 시장·조건 제거·비용 배수·추가 지연은 이 실행에서 평가하지 않았다.\n\n'
        (destination / 'REPORT.md').write_text(
            '# 고정 진입 대기 후보의 후속 비교\n\n' + scope_note + table(summary) + '\n\n'
            '활동·방향 모형, 대기 폭·만료 시간과 위험 설정을 다시 선택하지 않았다. '
            + ('unfiltered_v14는 같은 누적 빈도·관리 모델에서 반전만 평탄 청산으로 되돌린 원래 v14다. ' if frozen['protocol'] == 'minute_rate_reverse_v18' and not diagnostic_only else '')
            + ('unfiltered_v14는 원래 v14이며 inventory_fixed_size는 같은 새 수량 모델에서 축소 크기만 원래 고정값으로 바꾼 대조다. 부분 체결 총량의 다음 거래 가능 시가 실행 근사는 유지된다. ' if frozen['protocol'] in {'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21'} and not diagnostic_only else '')
            + ('previous_v20는 학습 시기와 원본 모델을 보존하고 추가 분봉 입력이 없는 대조다. ' if frozen['protocol'] in {'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24'} and not diagnostic_only else '')
            + ('unfiltered_v21은 추가 순효과 필터를 제거한 원래 v21이다. ' if frozen['protocol'] == 'addition_effect_v22' and not diagnostic_only else '')
            + ('previous_v21·previous_v20·unfiltered_v14·ungated_v7은 원래 진입 모델을 유지한 대조다. ' if frozen['protocol'] == 'recent_entry_v23' and not diagnostic_only else '')
            + (('unfiltered_v14·ungated_v7은 원래 활동 관문을 보존한 정책이다. unfiltered_direction_only는 새 기회 집합에서 순손익 필터만 제거한다. '
                if frozen['protocol'] == 'lifecycle_edge_v17' else 'unfiltered_v14는 같은 관리·위험 설정에서 순손익 진입 필터만 제거한 대조다. ') if is_edge and not diagnostic_only else '')
            + ('' if diagnostic_only else 'ungated_v7은 기존 30분·추가 금지 정책이며 no_adds와 cap_30m은 관리 정책의 해당 기능만 제한한다. '
               '진입 규칙이 같아도 보유·위험 상태 때문에 실제 진입 시각은 달라진다. ' if is_lifecycle else
               'ungated_v7은 순손익 필터만 제거한 기존 대기 정책이다. ' if is_net else
               'immediate는 같은 30분 보유와 위험 설정에서 대기만 제거한다. ')
            + ('기본 비용과 다음 시가 체결 조건을 유지했다.\n\n' if diagnostic_only else
               '비용은 편도 수수료·슬리피지를 함께 2·3배로 늘렸다. '
               '추가 지연은 1분이며, 조건 충족 후 실제 체결 가격은 다음 시가와 슬리피지로 계산했다.\n\n')
            +
            '대기 시작·충족·만료·취소와 실제 체결까지의 가격 변화는 waiting.json과 각 waiting_episodes.parquet에 보존했다. '
            '가격 변화는 왕복 비용을 차감한 실현 수익률과 다르다. 거래·비용·펀딩·최종 잔고 회계를 모두 대조했다.\n\n'
            '해당 기간의 연도별 초기화는 재학습이나 연속 운용이 아니다. 30일 블록의 95% 분위 구간은 관찰 경로에 조건부이며 '
            '여러 후보를 선택한 불확실성이나 시장 구조 변화를 포함하지 않는다. 현금과 영구 중지 뒤 기간도 포함한다. '
            '2026년 9월까지 이미 관찰한 구간이며 새 최종 평가·수익성 승인·실거래 준비 완료를 뜻하지 않는다.\n', encoding='utf-8')
        save_json(destination / 'summary.json', {'complete': True, 'period': period, 'symbols': symbols, 'decision': decision,
            'diagnostic_only': diagnostic_only, 'all_variant_conditions_completed': not diagnostic_only,
            'condition_runs': len(rows), 'annual_runs': len(annual)})
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        plt.close(chart)
    return destination
