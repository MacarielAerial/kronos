from enum import Enum


class NodeAttrKey(str, Enum):
    ntype = "ntype"
    raw_text = "raw_text"
    ner = "ner"


class EdgeAttrKey(str, Enum):
    etype = "etype"
    direction = "direction"
    distance = "distance"


class NodeType(str, Enum):
    year = "Year"
    month = "Month"
    day_of_week = "DayOfWeek"
    event = "Event"


class EdgeType(str, Enum):
    layout = "Layout"

    # Date to Date edge types
    year_to_month = "YearToMonth"
    month_to_day_of_week = "MonthToDayOfWeek"
    day_of_week_to_event = "DayOfWeekToEvent"
