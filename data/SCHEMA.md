# UFC Fight Data Schema

This document explains the structure of the fight data in `fight_snapshots.parquet` and `sample_fight.json`.

## Dataset Overview

The parquet file contains UFC fight records with:
- Fight metadata (fighters, date, outcome, method)
- Fighter stats at time of fight
- Complete fight history for both fighters leading up to this fight
- Round-by-round breakdowns for all historical fights

## Top-Level Fight Fields

### Fight Identification
- `fight_id` - Unique identifier for this fight
- `event_id` - Event where fight took place
- `fight_date` - Date of the fight (YYYY-MM-DD)
- `fight_link` - UFC Stats URL

### Fighters
- `fighter1_id` / `fighter2_id` - Fighter identifiers
- `fighter1_name` / `fighter2_name` - Fighter names
- `winner_id` / `loser_id` - Winner and loser identifiers

### Fight Details
- `method` - How fight ended (e.g., "d_unan" = decision unanimous)
- `fight_format` - Number of scheduled rounds (3 or 5)
- `fight_type` - Type of fight (e.g., "title")
- `referee` - Referee name
- `end_time` - Time fight ended (MM:SS)
- `weight_class` - Weight class

### Fighter Stats (at time of this fight)

#### Fighter 1 (f1_*)
- `f1_height_in` - Height in inches
- `f1_weight_lbs` - Weight in pounds
- `f1_reach_in` - Reach in inches
- `f1_stance` - Fighting stance (Orthodox/Southpaw)
- `f1_dob` - Date of birth
- `f1_win` / `f1_loss` - Record
- `f1_slpm` - Significant strikes landed per minute
- `f1_str_acc` - Striking accuracy (0-1)
- `f1_sapm` - Significant strikes absorbed per minute
- `f1_str_def` - Strike defense (0-1)
- `f1_td_avg` - Takedowns per fight
- `f1_td_acc` - Takedown accuracy (0-1)
- `f1_td_def` - Takedown defense (0-1)
- `f1_sub_avg` - Submissions per fight

#### Fighter 2 (f2_*)
Same fields as Fighter 1, prefixed with `f2_`

### Fight History Counts
- `prior_cnt_f1` - Number of previous fights for fighter 1
- `prior_cnt_f2` - Number of previous fights for fighter 2

## Prior Fight History Arrays

Each fighter has a `prior_f1` or `prior_f2` array containing all their previous fights in chronological order.

### Prior Fight Fields

#### Basic Info
- `fight_id` - Identifier for this prior fight
- `fight_date` - Date of prior fight
- `opponent_id` - Opponent identifier
- `result` - "win" or "loss"
- `method` - How fight ended
- `weight_class` - Weight class
- `end_time` - Time fight ended
- `fight_format` - Number of rounds

#### Fighter Stats (for this prior fight)
- `kd` - Knockdowns
- `sig_str_landed` / `sig_str_attempts` - Significant strikes
- `total_str_landed` / `total_str_attempts` - Total strikes
- `td_landed` / `td_attempts` - Takedowns
- `sub_att` - Submission attempts
- `rev` - Reversals
- `ctrl_time_s` - Control time in seconds
- `head_landed` / `head_attempts` - Head strikes
- `body_landed` / `body_attempts` - Body strikes
- `leg_landed` / `leg_attempts` - Leg strikes
- `distance_landed` / `distance_attempts` - Distance strikes
- `clinch_landed` / `clinch_attempts` - Clinch strikes
- `ground_landed` / `ground_attempts` - Ground strikes

#### Opponent Stats (opp_*)
Same stat fields prefixed with `opp_` showing what the opponent did in that fight.

#### Rounds Array
Each prior fight contains a `rounds` array with round-by-round breakdowns:

```json
{
  "fighter_id": "07f72a2a7591b409",
  "round_number": 1.0,
  "kd": 0.0,
  "sig_str_landed": 28.0,
  "sig_str_attempts": 39.0,
  "total_str_landed": 28.0,
  "total_str_attempts": 39.0,
  "td_landed": 0.0,
  "td_attempts": 3.0,
  "sub_att": 0.0,
  "rev": 0.0,
  "ctrl_time_s": 5.0,
  "head_landed": 5.0,
  "head_attempts": 12.0,
  "body_landed": 8.0,
  "body_attempts": 9.0,
  "leg_landed": 15.0,
  "leg_attempts": 18.0,
  "distance_landed": 27.0,
  "distance_attempts": 38.0,
  "clinch_landed": 1.0,
  "clinch_attempts": 1.0,
  "ground_landed": 0.0,
  "ground_attempts": 0.0
}
```

**Important:** The `fighter_id` field identifies whose stats these are (both fighters' rounds are in the same array).

## Sample Fight

The `sample_fight.json` file contains Jon Jones vs Daniel Cormier from 2015-01-03. This is a good reference for exploring the nested structure:

- 5-round title fight
- Both fighters have extensive prior history (30+ and 18+ fights)
- Each prior fight includes complete round breakdowns
- Shows both fighter and opponent perspectives

## Using the Data

### Reading the Parquet

**Polars (recommended):**
```python
import polars as pl
df = pl.read_parquet("data/fight_snapshots.parquet")
```

**Pandas:**
```python
import pandas as pd
df = pd.read_parquet("data/fight_snapshots.parquet")
```

### Exploring Nested Fields

**Polars:**
```python
# View one fight's prior history
fight = df.row(0, named=True)
prior_fights = fight['prior_f1']

# View rounds for a specific prior fight
rounds = prior_fights[0]['rounds']

# Access nested struct fields
df.select(pl.col("prior_f1").list.first().struct.field("fight_date"))
```

**Pandas:**
```python
# View one fight's prior history
fight = df.iloc[0]
prior_fights = fight['prior_f1']

# View rounds for a specific prior fight
rounds = prior_fights[0]['rounds']
```

### Filtering by Fighter ID
```python
# Rounds for specific fighter in a prior fight
fighter_id = "07f72a2a7591b409"
fighter_rounds = [r for r in rounds if r['fighter_id'] == fighter_id]
```
