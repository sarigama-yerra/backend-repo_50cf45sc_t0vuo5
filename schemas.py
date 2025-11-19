"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any

class FitnessProfile(BaseModel):
    name: str = Field(..., description="Full name")
    age: int = Field(..., ge=10, le=100)
    gender: Literal["Male", "Female", "Other"]
    height_cm: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)
    goal: Literal["Weight Loss", "Muscle Gain", "Maintenance", "Endurance", "Flexibility"]
    level: Literal["Beginner", "Intermediate", "Advanced"]
    location: Literal["Home", "Gym", "Outdoor"]
    diet: Literal["Veg", "Non-Veg", "Vegan", "Keto", "Paleo", "Any"]
    medical_history: Optional[str] = None
    stress_level: Optional[Literal["Low", "Medium", "High"]] = None

class DayWorkout(BaseModel):
    day: str
    exercises: List[Dict[str, Any]]

class DayMeals(BaseModel):
    day: str
    meals: List[Dict[str, Any]]

class Plan(BaseModel):
    profile: FitnessProfile
    workout: List[DayWorkout]
    diet: List[DayMeals]
    tips: List[str]

class SavedPlan(BaseModel):
    profile: FitnessProfile
    plan: Plan
    notes: Optional[str] = None
