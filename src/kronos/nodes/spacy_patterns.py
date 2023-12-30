#
# Reusable patterns for the spacy matcher
#
days_of_week = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "weds",
    "thurs",
    "sat",
    "mondays",
]
months = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
    "sept",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

#
# spacy patterns
#

YEAR_PATTERN = [{"SHAPE": "dddd"}]

MONTH_PATTERN = [{"LOWER": {"IN": months}}]

DAY_OF_WEEK_PATTERN = [{"LOWER": {"IN": days_of_week}}]

MONTH_DAY_PATTERN = [
    [  # e.g. Then april 4/5th
        {"LOWER": "then", "OP": "?"},  # Optionally matches "then" token
        {"LOWER": {"IN": months}},
        {"TEXT": {"REGEX": r"\d+(/\d+)?(st|nd|rd|th)?"}},  # e.g. 4/5th
    ],
    [
        {"LOWER": {"IN": months}},
        {
            "TEXT": {"REGEX": r"\d+(st|nd|rd|th)?"}
        },  # Matches numbers followed by optional ordinal suffix
    ],
]

DAY_MONTH_PATTERN = [
    [  # e.g. 7/8 th feb
        {
            "TEXT": {"REGEX": r"\d+(/\d+)?"}
        },  # Matches numbers and optional "/number", e.g., "7/8"
        {
            "TEXT": {"REGEX": r"(st|nd|rd|th)?"},
            "OP": "?",
        },  # Optionally matches ordinal suffix, e.g., "th"
        {"LOWER": {"IN": months}},
    ],
    [
        {
            "LOWER": "thurs",
            "OP": "?",
        },  # Optionally matches "then" token "thurs 16 march"
        {
            "TEXT": {"REGEX": r"\d+(st|nd|rd|th)?"},
            "OP": "?",
        },  # e.g. 18th or 25th of July
        {"LOWER": "or", "OP": "?"},  # Matches optional "or" token
        {
            "TEXT": {"REGEX": r"\d+(st|nd|rd|th)?"}
        },  # Matches numbers followed by optional ordinal suffix
        {"LOWER": "of", "OP": "?"},  # Matches optional "of" token
        {"LOWER": {"IN": months}},
    ],
    [  # e.g. 22JUL
        {
            "TEXT": {"REGEX": r"\d{1,2}[A-Z]{3}"}
        }  # Matches 1 or 2 digits followed by 3 uppercase letters
    ],
]

DURATION_PATTERN = [
    [{"LOWER": "term"}, {"LIKE_NUM": True}],  # e.g. term 1
    [
        {"LIKE_NUM": True},  # e.g. one week off or one week
        {"LOWER": {"IN": ["week", "weeks"]}},
        {"LOWER": "off", "OP": "?"},
    ],
    [
        {"IS_DIGIT": True, "OP": "{1,2}"},  # Matches any token that resembles a number
        {"LOWER": "full", "OP": "?"},  # Matches optional "full" token
        {
            "LOWER": {
                "IN": [
                    "days",
                    "weeks",
                    "months",
                    "years",
                    "hours",
                    "minutes",
                    "seconds",
                    "day",
                    "week",
                    "month",
                    "year",
                    "hour",
                    "minute",
                    "second",
                ]
            }
        },
    ],
    [  # e.g. 2 x 6 weeks
        {"LIKE_NUM": True},  # Matches any number
        {"LOWER": "x"},  # Matches multipliers
        {"LIKE_NUM": True},  # Matches another number
        {
            "LOWER": {
                "IN": [
                    "day",
                    "week",
                    "month",
                    "year",
                    "hour",
                    "minute",
                    "second",
                    "days",
                    "weeks",
                    "months",
                    "years",
                    "hours",
                    "minutes",
                    "seconds",
                ]
            }
        },  # Matches time units
    ],
]

DAY_OF_MONTH_PATTERN = [
    {
        "LOWER": {"IN": ["thurs", "sat"]},
        "OP": "?",
    },  # Optionally matches "thurs" before the number
    {
        "TEXT": {"REGEX": r"\d+(st|nd|rd|th|th)"}
    },  # Matches numbers followed by a compulsory ordinal suffix
    {
        "LOWER": {"IN": ["thurs", "sat"]},
        "OP": "?",
    },  # Optionally matches "thurs" after the number
]

TIME_PATTERN = [
    {"SHAPE": "dd:dd"},  # Matches the format of a time, e.g., "20:30"
    {"TEXT": "-"},  # Matches the dash separator
    {"SHAPE": "dd:dd"},  # Matches the format of a time, e.g., "21:45"
]

# Wack-a-mole patterns
KNOWN_PERSON_PATTERN = [{"LOWER": {"IN": ["will", "sharna"]}}]

KNOWN_LOC_PATTERN = [{"LOWER": "bar"}, {"LOWER": "salsa"}]
SPECIAL_DATE_PATTERN = [
    [
        {"LOWER": "the"},  # e.g. the following week
        {"LOWER": "following"},
        {"LOWER": "week"},
    ],
    [{"LOWER": "christmas"}],
    [
        {"LOWER": "end"},  # e.g. end week (as in the last week)
        {"LOWER": "week"},
    ],
    # e.g. LAUNCH BACHATA NIGHT
    [{"LOWER": "launch"}, {"LOWER": "bachata"}, {"LOWER": "night"}],
    # e.g. end of term
    [{"LOWER": "end"}, {"LOWER": "of"}, {"LOWER": "term"}],
]

# Anything that fits this pattern should be removed as an entity
REMOVE_PATTERN = [
    [
        {"LOWER": "should"},
        {"LOWER": "be"},
    ],
]
