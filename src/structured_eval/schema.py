from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Seniority(str, Enum):
    intern = "intern"
    entry = "entry"
    mid = "mid"
    senior = "senior"
    staff = "staff"
    manager = "manager"
    director = "director"
    executive = "executive"
    unknown = "unknown"


class EmploymentType(str, Enum):
    full_time = "full_time"
    part_time = "part_time"
    contract = "contract"
    internship = "internship"
    temporary = "temporary"
    unknown = "unknown"


class RemotePolicy(str, Enum):
    onsite = "onsite"
    hybrid = "hybrid"
    remote = "remote"
    unknown = "unknown"


class JobPostingLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    seniority: Seniority = Field(default=Seniority.unknown)
    employment_type: EmploymentType = Field(default=EmploymentType.unknown)
    location: Optional[str] = Field(default=None)
    remote_policy: RemotePolicy = Field(default=RemotePolicy.unknown)
    salary_min: Optional[int] = Field(default=None, ge=0)
    salary_max: Optional[int] = Field(default=None, ge=0)
    required_years_experience: Optional[float] = Field(default=None, ge=0)
    required_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    security_clearance_required: bool = Field(default=False)
    sponsorship_available: Optional[bool] = Field(default=None)
    labeling_notes: Optional[str] = Field(default=None)

    @field_validator("required_skills", "nice_to_have_skills")
    def skill_entries_must_not_be_blank(cls, value):
        if any(not skill.strip() for skill in value):
            raise ValueError("Skill entries must not be blank")
        return value

    @field_validator("salary_max")
    def salary_max_not_less_than_min(cls, value, info):
        if value is not None:
            salary_min = info.data.get("salary_min")
            if salary_min is not None and value < salary_min:
                raise ValueError("salary_max cannot be less than salary_min")
        return value
