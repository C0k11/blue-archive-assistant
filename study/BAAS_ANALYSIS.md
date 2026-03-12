# BAAS (Blue Archive Auto Script) Architecture Analysis

## Source: https://github.com/pur1fying/blue_archive_auto_script

## Core Architecture

### Detection Pattern: `co_detect` State Machine
BAAS uses a unified detection loop called `picture.co_detect()`:
- **img_possibles**: dict of {template_image_name: click_position} — if this template is visible, click the position
- **img_ends**: template(s) that signal "we've reached the target screen"
- **rgb_possibles**: dict of {color_feature_name: click_position} — pixel color checks
- **rgb_ends**: color feature(s) that signal arrival

This is a **declarative navigation system**: define what to click when you see X, and what the goal screen looks like. The engine loops screenshot→match→click→screenshot until it reaches an end state. Very robust — handles arbitrary popup chains automatically.

### Screen Resolution
- Fixed 1280x720 resolution (16:9)
- All coordinates are absolute pixel positions at this resolution
- Templates are pre-cropped at 1280x720

### Our Approach vs BAAS
| Aspect | BAAS | Ours |
|--------|------|------|
| Detection | Template matching + RGB pixel checks | OCR + YOLO + Florence + Template |
| Coordinates | Absolute pixels (1280x720) | Normalized 0-1 (resolution independent) |
| Navigation | Declarative co_detect state machine | Imperative skill tick functions |
| Popups | Image templates for each popup | OCR text matching |
| Multi-server | Separate templates per server (CN/Global/JP) | OCR-based (language agnostic) |

---

## Module-by-Module Analysis

### 1. Cafe (cafe_reward.py) — 602 lines

**Flow:**
1. Navigate to cafe → collect earnings → use invite ticket → headpat students → switch to cafe 2
2. Repeat invite + headpat on cafe 2

**Headpat Detection (method3 — current best):**
1. `zoom_out()` — pinch to zoom out the cafe view
2. Switch to gift view → swipe gift bar → screenshot mid-swipe → match happy faces
3. Uses 4 template images (`happy_face1.png` to `happy_face4.png`) with `cv2.matchTemplate` at threshold 0.75
4. Click each detected face position (offset +58 pixels down from match)
5. Repeat for `cafe_reward_affection_pat_round` rounds (configurable)

**Key Insight:** They swipe the gift bar to trigger student reaction animations, THEN screenshot to find the interaction bubbles. This is much more reliable than our YOLO approach because the bubbles are always visible after the swipe trigger.

**Invite Logic:**
- 3 invite methods: `lowest_affection`, `highest_affection`, `starred`, or `target_student` (by name)
- Target student: OCR each student name in invite list, scroll through pages, match against configured name
- Sort order: can sort by affection/name/academy/starred
- Handles duplicate invite, clothes switch, and invalid ticket states

**What We Should Copy:**
- Gift bar swipe → screenshot → template match pattern for headpat
- Multiple invite strategies (lowest/highest affection, specific student)
- Duplicate invite handling
- Zoom out before scanning

### 2. Schedule/Lesson (lesson.py) — 618 lines

**Flow:**
1. Check ticket count (purchase tickets if configured)
2. If `lesson_enableInviteFavorStudent`: scan ALL regions to find specified students
3. For each configured region: open 全體課程表, detect status of all 9 rooms, choose best room, execute

**Favor Student Search (KEY FEATURE):**
1. Load avatar template images for each favorite student
2. Iterate through ALL regions (switch pages left/right)
3. On each region's 全體課程表: for each available room, search for student avatar templates
4. Build a map: `{student_name: set of (region, block_id)}`
5. First student found → execute immediately
6. Remaining students → use the recorded positions to jump directly

**Room Status Detection:**
- 9 rooms in 3x3 grid at fixed positions
- Check RGB pixel color at each room's status indicator:
  - White (250-255) = available
  - Grey (230-249) = done
  - Dark (31-160) = locked
  - Medium grey (197-217) = no activity

**Room Choice Logic (`choose_lesson`):**
- `lesson_relationship_first`: pick room with most affection hearts (maximize relationship gain)
- Otherwise: follow tier priority (superior > advanced > normal > primary)
- Within same tier: prefer higher relationship count

**Room Click Positions (全體課程表 popup):**
```python
click_lo = [[307, 257], [652, 257], [995, 257],
            [307, 408], [652, 408], [995, 408],
            [307, 560], [652, 560], [985, 560]]
```
These are the 9 room card positions in the popup grid.

**What We Should Copy:**
- Template matching for student avatars across ALL regions
- Room status detection via pixel color (available/done/locked)
- Relationship count detection via RGB pixel checks
- Tier-based room priority with relationship tiebreaker
- Fixed click positions for the 9-room grid in 全體課程表

### 3. Arena (arena.py) — 174 lines

**Flow:**
1. Navigate to tactical challenge
2. Get ticket count via OCR
3. If `ArenaStopFightWhenRank1`: check current rank, skip if rank 1
4. Choose enemy based on level diff (refresh if opponent too strong)
5. Fight with skip enabled
6. If >1 ticket: set next_time=55s (re-run after cooldown)
7. Collect time reward + daily reward

**Enemy Selection:**
- OCR self level and opponent level
- If opponent too high (exceeds `ArenaLevelDiff`), refresh
- Max refresh times configurable

**Reward Collection:**
- Check RGB color at reward button positions
- Green (235-255, 222-242, 52-92) = collectible
- Grey (206-226) = already collected

**What We Should Copy:**
- Level-based opponent filtering with refresh
- Separate time reward and daily reward collection
- Re-run with timer for multiple tickets

### 4. Craft/Manufacturing (create.py) — 1029 lines

**Flow:**
1. Check crafting slot status (empty/unfinished/done)
2. Collect finished items
3. For each empty slot: create new item with up to 3 phases
4. Use acceleration tickets if configured
5. Track daily creation count

**3-Phase Crafting:**
- Phase 1: select base materials
- Phase 2: add more materials (if max_phase >= 2)
- Phase 3: final materials (if max_phase >= 3)
- Confirm and start crafting

**What We Should Copy:**
- 3-slot management (check each slot independently)
- Phase-based crafting with configurable max phase
- Acceleration ticket usage
- Unfinished craft recovery
- Daily creation count tracking

### 5. MomoTalk (momo_talk.py) — 184 lines

**Flow:**
1. Navigate to MomoTalk
2. Sort by "unread" and "descending"
3. Detect unread message positions via color/template
4. Click each unread → auto-complete story/dialogue
5. Restart from main page to catch any new unreads

**Story Completion:**
- Uses `common_solve_affection_story_method` — clicks through dialogue/cutscenes
- Handles multiple story types (affection, main, group)

**What We Should Copy:**
- Sort by unread for efficient processing
- Auto-complete all dialogue/story
- Re-check after completing (new messages may appear)

### 6. Total Assault (total_assault.py) — 423 lines

**Flow:**
1. Get ticket count
2. Check for unfinished fights → finish them
3. Find highest available difficulty
4. Fight: if WIN, try next difficulty up; if LOSE, go down
5. Once highest beatable found: sweep remaining tickets
6. Collect season reward + accumulated point reward

**Difficulty Progression:**
- NORMAL → HARD → VERYHARD → HARDCORE → EXTREME → INSANE → TORMENT
- Binary search by fighting: win=go up, lose=go down
- Sweep at highest beatable level

**What We Should Copy:**
- Difficulty auto-detection via fight results
- Sweep for remaining tickets after calibration
- Season + accumulated reward collection

### 7. Shop (shop/ folder)

**Features:**
- Normal shop: buy specified items
- Arena shop: auto-buy and configurable refresh count
- AP purchase with configurable count

---

## Implementation Priority for Our Pipeline

### Phase 1: Fix Current Broken Features (HIGH)
1. **Schedule**: Copy BAAS's room click positions for 全體課程表 grid
2. **Schedule**: Copy room status detection via pixel color (available/done)
3. **Schedule**: Copy relationship-first room choice logic
4. **Cafe headpat**: Consider BAAS's gift-bar swipe method as alternative to YOLO

### Phase 2: Add Missing Features (HIGH)
5. **MomoTalk**: Auto-complete all unread conversations
6. **Total Assault**: Use all tickets + collect rewards
7. **Tactical Exam**: Auto-clear during exam period (joint_firing_drill.py)
8. **Arena**: Level-based opponent selection, multi-ticket with timer

### Phase 3: Enhance Existing Features (MEDIUM)
9. **Cafe invite**: Add affection-based sorting (lowest/highest)
10. **Schedule**: Template-match favorite students across ALL regions
11. **Craft**: 3-phase crafting with slot management
12. **Shop**: Arena shop auto-buy

### Phase 4: Advanced Features (LOW)
13. **Campaign**: Auto-push story mode
14. **Activity**: One-click event story + challenge + mission
15. **AP Planning**: Smart AP usage during 2x/3x drop events
16. **Arena stone mining**: New season gem farming

---

## Key Architectural Improvements

### 1. Declarative Navigation (co_detect pattern)
Our imperative tick-based approach leads to many edge cases. Consider adding a `navigate_to(target_screen, click_map)` utility that loops until reaching the target, clicking appropriate buttons along the way.

### 2. Room Status via Pixel Color
BAAS detects schedule room availability by checking pixel color at fixed positions. This is more reliable than our OCR-based approach. We should add similar pixel color checks for:
- Schedule room available/done/locked
- Arena reward collectible/collected
- Invitation ticket available/cooldown

### 3. Fixed Position Click Maps
BAAS stores click positions for grid-based UIs (schedule rooms, shop items, etc.) as hardcoded arrays at 1280x720. We should maintain similar maps but in normalized 0-1 coordinates.

### 4. Template Matching for Student Avatars
BAAS uses cropped avatar templates to find specific students in schedule. We use Florence which is heavier. For known students, pre-cropped templates would be faster and more reliable.
