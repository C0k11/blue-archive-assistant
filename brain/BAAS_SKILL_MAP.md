# BAAS → Pipeline Skill Mapping

> Reference for adapting BAAS click sequences/logic into our pipeline skills.
> BAAS source: `study/baas/module/`  |  Our skills: `brain/skills/`

## Core BAAS Patterns

BAAS uses `picture.co_detect(self, rgb_ends, rgb_possibles, img_ends, img_possibles)`:
- **img_possibles**: template→click-coordinate dict. If template matches, click that coordinate.
- **img_ends**: target template(s). Loop ends when one of these matches.
- **rgb_possibles/ends**: pixel-color checks at specific coordinates.
- All coordinates are in **1280×720** absolute pixels.

Our pipeline uses **normalized [0,1] coordinates** and OCR-based detection.
Conversion: `nx = px / 1280`, `ny = py / 720`.

---

## Skill Mapping Table

| Our Skill | BAAS Module | Status | Key Differences |
|-----------|-------------|--------|------------------|
| `lobby` | `to_main_page()` (built-in) | ✅ Working | BAAS uses template; we use OCR |
| `event_activity` | `module/activities/*` | ✅ Working | Event-specific modules per event |
| `event_farming` | `explore_tasks/sweep_task` | ✅ Working | BAAS uses stage JSON data |
| `cafe` | `cafe_reward` | ✅ Working | BAAS uses template-match for happy faces |
| `schedule` | `lesson` | ✅ Working | BAAS has 9-slot grid detection via RGB |
| `club` | `group` | ✅ Working | 30-line module: main_page→group nav→sign-up-reward |
| `craft` | `create` | ✅ Working | 1029-line module: 3-phase crafting with node OCR |
| `daily_tasks` | `collect_daily_task_power` | ✅ Working | RGB button-color check for claim/grey/complete |
| `mail` | `mail` | ✅ Working | 28-line: main_page→mail→RGB check collect button |
| `arena` | `arena` + `scrimmage` | ✅ Working | Opponent level OCR + configurable refresh |
| `bounty` | `rewarded_task` | ✅ Working | 3-location sweep with ticket purchase |
| `total_assault` | `total_assault` | ✅ Working | Multi-formation fight + difficulty detection |
| `joint_firing_drill` | `joint_firing_drill` | ✅ Working | Similar to total_assault |
| `ap_planning` | `purchase_ap` + `collect_daily_free_power` | ✅ Working | AP collection + optional purchase |
| `campaign_push` | `explore_tasks` | ✅ Working | Stage progression |
| `pass_reward` | `collect_pass_reward` | ✅ Working | JP-only: mission pts + pass rewards + statistics |
| `momo_talk` | `momo_talk` | 🔧 Fixed | Tab nav, BAAS icon pos, broader detection |
| `shop` | `shop/common_shop` | 🔧 Fixed | Purchase loop termination, attempt tracking |
| `story_cleanup` | `main_story` | 🔧 Fixed | Hub loop detection, tighter idle thresholds |

---

## Applied Fixes (BAAS-Adapted Logic)

### 1. MomoTalk (`momo_talk.py`) — FIXED

**Root cause**: Wrong sidebar icon position + no tab navigation to conversations tab.

**Fixes applied**:
- Corrected sidebar icon position to BAAS-verified (0.130, 0.208)
- Added tab navigation: click conversations tab (0.131, 0.281) after entry
- Broadened `_is_momotalk` detection: sort controls as fallback
- Widened OCR search region for sidebar icon text
- Added stall recovery: retry icon click after 5 ticks of no detection
- Added reply-visible detection during scan (catches already-open conversations)
- Lowered OCR confidence thresholds for unread badges

### 2. Shop (`shop.py`) — FIXED

**Root cause**: Purchase loop never terminated when confirm dialog OCR missed.

**Fixes applied**:
- Added `_buy_btn_clicks` counter — exits after 3 attempts without confirm
- Added `_confirm_clicks` counter for proper post-confirm timeout
- Broader confirm dialog detection after buy button click (wider region, lower conf)
- Earlier exit when no purchase bar found after select-all
- Reduced purchase timeout from 20 to 15 ticks
- Lowered OCR confidence for bulk buy button detection

### 3. StoryCleanup (`story_cleanup.py`) — FIXED

**Root cause**: Hub detected but no actionable nodes found; idle threshold too generous.

**Fixes applied**:
- Added `_hub_consecutive` counter — exits section after 4 hub views with 0 actions
- Added `_entered_episode_list` flag to gate actionable text search
- Episode node search now restricted to y>0.30 to avoid hub tab text matches
- Removed 確認/确定 from non-dialogue next-button (prevents false popup captures)
- Tightened idle thresholds: hub idle 5→3, section timeout 40→30, abs idle 8→6
- Reset hub/episode tracking per section

---

## BAAS Key Coordinates (1280×720 → normalized)

### Navigation
| Element | BAAS px | Normalized |
|---------|---------|------------|
| Main page home | (95, 699) | (0.074, 0.971) |
| Main page bus | (1098, 261) | (0.858, 0.363) |
| Cafe menu | (887, 647) | (0.693, 0.899) |
| Shop menu | (1163, 659) | (0.909, 0.915) |
| Lesson location | (210, 655) | (0.164, 0.910) |

### Common Popups
| Popup | BAAS click | Normalized |
|-------|-----------|------------|
| Reward acquired | (640, 116) | (0.500, 0.161) |
| Relationship rank up | (640, 360) | (0.500, 0.500) |
| Full notice | (887, 165) | (0.693, 0.229) |
| Notice dismiss | (887, 166) | (0.693, 0.231) |

---

## BAAS Module Complexity Reference

| Module | Lines | Pattern | Notes |
|--------|-------|---------|-------|
| `group.py` | 30 | Simple co_detect chain | Nav→enter→sign-up-reward |
| `mail.py` | 28 | RGB button check | Collect if yellow, skip if grey |
| `collect_daily_task_power.py` | 62 | RGB claim/grey detection | Two buttons: daily + pyroxene |
| `collect_pass_reward.py` | 147 | co_detect + OCR stats | JP-only, level/point/weekly OCR |
| `momo_talk.py` | 184 | RGB state machine | Tab nav, sort, unread scan, dialogue |
| `arena.py` | 174 | OCR level + template | Opponent level check, skip toggle |
| `rewarded_task.py` | 202 | RGB sweep detect | 3 bounty locations, ticket purchase |
| `common_shop.py` | 274 | RGB item + template | Position scanning, currency detection |
| `lesson.py` | 618 | OCR + RGB grid | 9-slot grid, region selection, invite |
| `cafe_reward.py` | 602 | Template + OCR | Collect, invite, headpat, attitude |
| `main_story.py` | 379 | Template + co_detect | Episode nav, plot skip, grid tasks |
| `total_assault.py` | 423 | OCR difficulty + RGB | Multi-formation, sweep, rewards |
| `create.py` | 1029 | OCR + template + RGB | 3-phase crafting, node priority, filters |
