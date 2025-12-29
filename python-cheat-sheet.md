---
id: python-cheatsheet
aliases:
  - python-cheatsheet
  - py-reference
tags:
  - python
  - cheatsheet
  - reference
created: 2025-12-26 12:50
modified: 2025-12-26 13:43
refs:
  - "[python](python.md)"
  - "[full-stack-development](full-stack-development.md)"
title: Python Comprehensive Cheat Sheet
topics:
  - python
  - pydantic
  - fastapi
  - async
  - agents
---

# Python Comprehensive Cheat Sheet

A practical reference for Python development, focused on building AI agents and APIs.

---

## Table of Contents

1. [Syntax Fundamentals](#syntax-fundamentals)
2. [Data Types](#data-types)
3. [Data Structures](#data-structures)
4. [Control Flow](#control-flow)
5. [Functions](#functions)
6. [Type Hints](#type-hints)
7. [Classes & OOP](#classes--oop)
8. [Decorators](#decorators)
9. [Context Managers](#context-managers)
10. [Generators & Iterators](#generators--iterators)
11. [Error Handling](#error-handling)
12. [File I/O](#file-io)
13. [Modules & Packages](#modules--packages)
14. [Async/Await](#asyncawait)
15. [Pydantic](#pydantic)
16. [FastAPI](#fastapi)
17. [Testing with Pytest](#testing-with-pytest)
18. [Agent SDK Patterns](#agent-sdk-patterns)
19. [Common Patterns](#common-patterns)
20. [Standard Library Gems](#standard-library-gems)

---

## Syntax Fundamentals

### Variables & Assignment
```python
# No type declaration needed (but type hints recommended)
name = "Claude"
age = 3
pi = 3.14159
is_active = True

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 0

# Swap values
x, y = y, x

# Unpacking
first, *rest = [1, 2, 3, 4]  # first=1, rest=[2,3,4]
first, *middle, last = [1, 2, 3, 4, 5]  # middle=[2,3,4]

# Walrus operator (assignment expression) - Python 3.8+
if (n := len(items)) > 10:
    print(f"Too many items: {n}")
```

### String Formatting
```python
name = "Agent"
version = 4.0

# f-strings (preferred)
message = f"Hello, {name} v{version}"
formatted = f"{3.14159:.2f}"  # "3.14"
padded = f"{42:05d}"  # "00042"
aligned = f"{name:>10}"  # "     Agent"

# Multiline f-strings
query = f"""
SELECT * FROM users
WHERE name = '{name}'
AND version >= {version}
"""

# Raw strings (no escape processing)
path = r"C:\Users\name\file.txt"

# Debug with f-strings (Python 3.8+)
x = 42
print(f"{x=}")  # prints "x=42"
```

### Comments & Docstrings
```python
# Single line comment

"""
Multiline string (not a comment, but often used as one)
"""

def greet(name: str) -> str:
    """
    Generate a greeting message.

    Args:
        name: The name to greet.

    Returns:
        A personalized greeting string.

    Raises:
        ValueError: If name is empty.

    Example:
        >>> greet("Claude")
        'Hello, Claude!'
    """
    if not name:
        raise ValueError("Name cannot be empty")
    return f"Hello, {name}!"
```

---

## Data Types

### Numeric Types
```python
# Integers (arbitrary precision)
x = 42
big = 10**100  # No overflow
binary = 0b1010  # 10
octal = 0o17  # 15
hexadecimal = 0xFF  # 255
with_underscores = 1_000_000  # Readability

# Floats
pi = 3.14159
scientific = 1.5e-10
infinity = float('inf')
not_a_number = float('nan')

# Complex
c = 3 + 4j
c.real  # 3.0
c.imag  # 4.0

# Type conversion
int("42")  # 42
float("3.14")  # 3.14
str(42)  # "42"
bool(0)  # False
bool(1)  # True
bool("")  # False
bool("any")  # True
```

### Boolean & None
```python
# Booleans
True, False

# Falsy values (evaluate to False)
False, None, 0, 0.0, "", [], {}, set(), ()

# Truthy (everything else)
True, 1, "text", [1], {"a": 1}

# None (like nil in Ruby, null in JS)
value = None
if value is None:  # Use 'is', not '=='
    print("No value")

# Ternary operator
result = "yes" if condition else "no"
```

### Strings
```python
s = "Hello, World!"

# Indexing & Slicing
s[0]      # 'H' (first)
s[-1]     # '!' (last)
s[0:5]    # 'Hello' (slice)
s[:5]     # 'Hello' (from start)
s[7:]     # 'World!' (to end)
s[::2]    # 'Hlo ol!' (every 2nd)
s[::-1]   # '!dlroW ,olleH' (reverse)

# Common methods
s.lower()               # 'hello, world!'
s.upper()               # 'HELLO, WORLD!'
s.title()               # 'Hello, World!'
s.strip()               # Remove whitespace
s.split(", ")           # ['Hello', 'World!']
", ".join(['a', 'b'])   # 'a, b'
s.replace("World", "Agent")  # 'Hello, Agent!'
s.startswith("Hello")   # True
s.endswith("!")         # True
s.find("World")         # 7 (index, -1 if not found)
s.count("l")            # 3
s.isdigit()             # False
s.isalpha()             # False (has punctuation)
s.isalnum()             # False

# String formatting
f"Value: {42}"
"Value: {}".format(42)
"Value: %d" % 42

# Multiline
multi = """Line 1
Line 2
Line 3"""

# Check substring
"World" in s  # True
```

---

## Data Structures

### Lists (Mutable Arrays)
```python
# Creation
nums = [1, 2, 3, 4, 5]
mixed = [1, "two", 3.0, [4, 5]]
empty = []
from_range = list(range(5))  # [0, 1, 2, 3, 4]

# Access
nums[0]     # 1 (first)
nums[-1]    # 5 (last)
nums[1:3]   # [2, 3] (slice)

# Modification
nums.append(6)          # Add to end
nums.insert(0, 0)       # Insert at index
nums.extend([7, 8])     # Add multiple
nums.remove(3)          # Remove first occurrence
nums.pop()              # Remove & return last
nums.pop(0)             # Remove & return at index
nums.clear()            # Remove all

# Operations
len(nums)               # Length
nums.index(3)           # Index of first 3
nums.count(3)           # Count occurrences
nums.sort()             # Sort in place
nums.sort(reverse=True) # Sort descending
nums.reverse()          # Reverse in place
sorted(nums)            # Return new sorted list
nums.copy()             # Shallow copy
list(reversed(nums))    # Return new reversed list

# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
pairs = [(x, y) for x in range(3) for y in range(3)]
flattened = [item for sublist in nested for item in sublist]

# Conditional comprehension
result = [x if x > 0 else 0 for x in numbers]

# Unpack
first, second, *rest = [1, 2, 3, 4, 5]

# Check membership
3 in nums  # True

# Concatenate
[1, 2] + [3, 4]  # [1, 2, 3, 4]

# Repeat
[0] * 5  # [0, 0, 0, 0, 0]
```

### Tuples (Immutable)
```python
# Creation
point = (3, 4)
single = (42,)  # Note the comma!
empty = ()
from_list = tuple([1, 2, 3])

# Access (same as list)
point[0]  # 3

# Unpacking
x, y = point

# Named tuples (better)
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(3, 4)
p.x  # 3
p.y  # 4

# Use cases
# - Function return multiple values
# - Dictionary keys (lists can't be keys)
# - Data that shouldn't change
```

### Dictionaries (Hash Maps)
```python
# Creation
user = {"name": "Claude", "age": 3}
empty = {}
from_pairs = dict([("a", 1), ("b", 2)])
from_keys = dict.fromkeys(["a", "b"], 0)  # {"a": 0, "b": 0}

# Access
user["name"]            # "Claude" (KeyError if missing)
user.get("name")        # "Claude" (None if missing)
user.get("email", "")   # "" (default if missing)

# Modification
user["email"] = "claude@anthropic.com"  # Add/update
user.update({"age": 4, "model": "opus"})  # Merge
del user["age"]         # Delete key
user.pop("age", None)   # Remove & return (with default)
user.setdefault("role", "assistant")  # Set if missing

# Iteration
user.keys()             # dict_keys(['name', 'age'])
user.values()           # dict_values(['Claude', 3])
user.items()            # dict_items([('name', 'Claude'), ...])

for key in user:
    print(key)

for key, value in user.items():
    print(f"{key}: {value}")

# Dict comprehension
squares = {x: x**2 for x in range(5)}
filtered = {k: v for k, v in user.items() if v}

# Merge (Python 3.9+)
merged = dict1 | dict2
dict1 |= dict2  # In-place merge

# Check key
"name" in user  # True

# Nested access
data = {"user": {"profile": {"name": "Claude"}}}
name = data.get("user", {}).get("profile", {}).get("name")

# defaultdict (auto-create missing keys)
from collections import defaultdict
counts = defaultdict(int)
counts["a"] += 1  # No KeyError

groups = defaultdict(list)
groups["team"].append("member")
```

### Sets (Unique Values)
```python
# Creation
nums = {1, 2, 3, 3, 3}  # {1, 2, 3}
empty = set()  # NOT {} (that's a dict)
from_list = set([1, 2, 2, 3])

# Modification
nums.add(4)
nums.remove(3)      # KeyError if missing
nums.discard(3)     # No error if missing
nums.pop()          # Remove arbitrary element
nums.clear()

# Operations
a = {1, 2, 3}
b = {2, 3, 4}

a | b   # Union: {1, 2, 3, 4}
a & b   # Intersection: {2, 3}
a - b   # Difference: {1}
a ^ b   # Symmetric difference: {1, 4}

a.issubset(b)       # Is a ⊆ b?
a.issuperset(b)     # Is a ⊇ b?
a.isdisjoint(b)     # No common elements?

# Set comprehension
evens = {x for x in range(10) if x % 2 == 0}

# Frozen set (immutable, can be dict key)
frozen = frozenset([1, 2, 3])
```

### Collections Module
```python
from collections import (
    Counter,
    defaultdict,
    OrderedDict,
    deque,
    namedtuple,
    ChainMap
)

# Counter - count occurrences
words = ["a", "b", "a", "c", "a", "b"]
counts = Counter(words)  # Counter({'a': 3, 'b': 2, 'c': 1})
counts.most_common(2)    # [('a', 3), ('b', 2)]
counts["a"]              # 3
counts.update(["a", "a"])  # Add more

# deque - double-ended queue (fast append/pop both ends)
q = deque([1, 2, 3])
q.append(4)          # Right
q.appendleft(0)      # Left
q.pop()              # Right
q.popleft()          # Left
q.rotate(1)          # Rotate right

# ChainMap - search multiple dicts
defaults = {"color": "red", "size": "medium"}
overrides = {"color": "blue"}
config = ChainMap(overrides, defaults)
config["color"]  # "blue"
config["size"]   # "medium"
```

---

## Control Flow

### Conditionals
```python
# if/elif/else
if x > 0:
    print("positive")
elif x < 0:
    print("negative")
else:
    print("zero")

# Ternary
result = "even" if x % 2 == 0 else "odd"

# Match statement (Python 3.10+)
match command:
    case "start":
        start()
    case "stop":
        stop()
    case ["move", x, y]:
        move(int(x), int(y))
    case {"action": action, "value": value}:
        handle(action, value)
    case _:
        print("Unknown command")

# Match with guards
match point:
    case (x, y) if x == y:
        print("On diagonal")
    case (x, y):
        print(f"Point at {x}, {y}")

# Truthiness checks
if items:          # True if not empty
    process(items)

if not errors:     # True if empty
    success()

# Chained comparisons
if 0 < x < 10:
    print("Single digit positive")
```

### Loops
```python
# for loop
for item in items:
    print(item)

for i in range(5):           # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 5):        # 2, 3, 4
    print(i)

for i in range(0, 10, 2):    # 0, 2, 4, 6, 8
    print(i)

for i in range(5, 0, -1):    # 5, 4, 3, 2, 1
    print(i)

# enumerate (index + value)
for i, item in enumerate(items):
    print(f"{i}: {item}")

for i, item in enumerate(items, start=1):  # Start from 1
    print(f"{i}: {item}")

# zip (parallel iteration)
names = ["Alice", "Bob"]
scores = [95, 87]
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# zip_longest (handle unequal lengths)
from itertools import zip_longest
for a, b in zip_longest(short, long, fillvalue=0):
    print(a, b)

# while loop
while condition:
    do_something()
    if should_stop:
        break
    if should_skip:
        continue

# else clause (runs if no break)
for item in items:
    if matches(item):
        break
else:
    print("No match found")

# Infinite loop
while True:
    if done:
        break
```

---

## Functions

### Basic Functions
```python
# Definition
def greet(name):
    return f"Hello, {name}!"

# With type hints (recommended)
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Default arguments
def greet(name: str = "World") -> str:
    return f"Hello, {name}!"

# *args (variable positional)
def sum_all(*args: int) -> int:
    return sum(args)

sum_all(1, 2, 3)  # 6

# **kwargs (variable keyword)
def create_user(**kwargs) -> dict:
    return kwargs

create_user(name="Claude", age=3)

# Combined
def func(pos1, pos2, /, pos_or_kw, *, kw_only):
    """
    pos1, pos2: positional only (before /)
    pos_or_kw: either positional or keyword
    kw_only: keyword only (after *)
    """
    pass

# Unpack arguments
args = [1, 2, 3]
kwargs = {"a": 1, "b": 2}
func(*args, **kwargs)
```

### Lambda Functions
```python
# Anonymous functions
double = lambda x: x * 2
add = lambda x, y: x + y

# Common uses
sorted(items, key=lambda x: x.name)
filtered = filter(lambda x: x > 0, numbers)
mapped = map(lambda x: x * 2, numbers)

# Prefer list comprehension over map/filter
[x * 2 for x in numbers if x > 0]
```

### Higher-Order Functions
```python
from functools import reduce, partial

# map - apply function to each element
list(map(str.upper, ["a", "b", "c"]))  # ["A", "B", "C"]

# filter - keep elements where function returns True
list(filter(lambda x: x > 0, [-1, 0, 1, 2]))  # [1, 2]

# reduce - accumulate
reduce(lambda acc, x: acc + x, [1, 2, 3, 4])  # 10

# partial - fix some arguments
def power(base, exp):
    return base ** exp

square = partial(power, exp=2)
cube = partial(power, exp=3)

# sorted with key
people = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
sorted(people, key=lambda p: p["age"])

from operator import itemgetter, attrgetter
sorted(people, key=itemgetter("age"))
sorted(objects, key=attrgetter("name"))
```

### Closures
```python
def make_multiplier(n):
    def multiply(x):
        return x * n
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)
double(5)  # 10
triple(5)  # 15

# nonlocal - modify enclosing scope variable
def counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

c = counter()
c()  # 1
c()  # 2
```

---

## Type Hints

### Basic Types
```python
from typing import (
    List, Dict, Set, Tuple,
    Optional, Union, Any,
    Callable, TypeVar, Generic,
    Literal, Final, TypedDict,
    Annotated
)

# Primitives
name: str = "Claude"
age: int = 3
score: float = 95.5
active: bool = True

# Collections (Python 3.9+ can use lowercase)
names: list[str] = ["Alice", "Bob"]
scores: dict[str, int] = {"Alice": 95}
unique: set[int] = {1, 2, 3}
point: tuple[int, int] = (3, 4)
mixed_tuple: tuple[str, int, float] = ("a", 1, 2.0)
any_length_tuple: tuple[int, ...] = (1, 2, 3, 4)

# Optional (can be None)
email: str | None = None  # Python 3.10+
email: Optional[str] = None  # Older syntax

# Union (multiple types)
id: int | str = 42  # Python 3.10+
id: Union[int, str] = 42  # Older syntax

# Any (escape hatch - avoid)
data: Any = something_unknown()

# Literal (specific values only)
status: Literal["active", "inactive", "pending"]

# Final (can't be reassigned)
MAX_SIZE: Final = 100
```

### Function Types
```python
# Basic function
def greet(name: str) -> str:
    return f"Hello, {name}"

# No return value
def log(message: str) -> None:
    print(message)

# Callable type
Handler = Callable[[str, int], bool]  # (str, int) -> bool

def register(handler: Handler) -> None:
    pass

# With defaults
def greet(name: str = "World") -> str:
    return f"Hello, {name}"

# *args and **kwargs
def func(*args: int, **kwargs: str) -> None:
    pass
```

### Generics
```python
from typing import TypeVar, Generic

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Generic function
def first(items: list[T]) -> T:
    return items[0]

# Generic class
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

# Usage
stack: Stack[int] = Stack()
stack.push(1)

# Bounded TypeVar
from typing import Comparable
T = TypeVar('T', bound=Comparable)
```

### TypedDict
```python
from typing import TypedDict, Required, NotRequired

class User(TypedDict):
    name: str
    age: int
    email: NotRequired[str]  # Optional key

# Total=False means all keys optional by default
class PartialUser(TypedDict, total=False):
    name: str
    age: int

user: User = {"name": "Claude", "age": 3}
```

### Protocol (Structural Subtyping)
```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

# Circle is a Drawable (no explicit inheritance needed)
def render(item: Drawable) -> None:
    item.draw()

render(Circle())  # Works!
```

---

## Classes & OOP

### Basic Class
```python
class User:
    """A user in the system."""

    # Class attribute (shared by all instances)
    species = "human"

    def __init__(self, name: str, age: int) -> None:
        """Initialize user."""
        self.name = name  # Instance attribute
        self.age = age
        self._private = "convention"  # "Private" by convention
        self.__mangled = "name mangled"  # Name mangling

    def greet(self) -> str:
        """Instance method."""
        return f"Hello, I'm {self.name}"

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Alternative constructor."""
        return cls(data["name"], data["age"])

    @staticmethod
    def validate_age(age: int) -> bool:
        """Static method (no self/cls)."""
        return 0 <= age <= 150

    @property
    def is_adult(self) -> bool:
        """Read-only property."""
        return self.age >= 18

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"User(name={self.name!r}, age={self.age})"

    def __str__(self) -> str:
        """User-friendly string."""
        return f"{self.name} ({self.age})"

# Usage
user = User("Claude", 3)
user.greet()
user.is_adult  # Property access (no parentheses)
User.from_dict({"name": "Claude", "age": 3})
```

### Dataclasses (Recommended for Data)
```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class User:
    name: str
    age: int
    email: Optional[str] = None
    tags: list[str] = field(default_factory=list)  # Mutable default

    def __post_init__(self):
        """Called after __init__."""
        self.name = self.name.strip()

# Auto-generates __init__, __repr__, __eq__
user = User(name="Claude", age=3)
print(user)  # User(name='Claude', age=3, email=None, tags=[])

# Frozen (immutable)
@dataclass(frozen=True)
class Point:
    x: float
    y: float

# With slots (memory efficient)
@dataclass(slots=True)
class Efficient:
    x: int
    y: int

# With ordering
@dataclass(order=True)
class Priority:
    priority: int
    name: str = field(compare=False)  # Exclude from comparison
```

### Inheritance
```python
class Animal:
    def __init__(self, name: str) -> None:
        self.name = name

    def speak(self) -> str:
        raise NotImplementedError

class Dog(Animal):
    def speak(self) -> str:
        return f"{self.name} says woof!"

class Cat(Animal):
    def speak(self) -> str:
        return f"{self.name} says meow!"

# Multiple inheritance
class FlyingMixin:
    def fly(self) -> str:
        return f"{self.name} is flying"

class Bird(Animal, FlyingMixin):
    def speak(self) -> str:
        return f"{self.name} says chirp!"

# Call parent method
class Child(Parent):
    def method(self):
        super().method()  # Call parent's method
```

### Abstract Base Classes
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        """Calculate area."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """Calculate perimeter."""
        pass

class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * 3.14159 * self.radius

# Can't instantiate Shape directly
# shape = Shape()  # TypeError
```

### Dunder Methods (Magic Methods)
```python
class Vector:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    # String representations
    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    # Comparison
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __lt__(self, other: "Vector") -> bool:
        return self.magnitude < other.magnitude

    # Arithmetic
    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> "Vector":
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector":
        return self * scalar  # 3 * vector

    # Container methods
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> float:
        return [self.x, self.y][index]

    def __iter__(self):
        yield self.x
        yield self.y

    # Callable
    def __call__(self, scale: float) -> "Vector":
        return self * scale

    # Context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # Hash (for use in sets/dict keys)
    def __hash__(self) -> int:
        return hash((self.x, self.y))

    @property
    def magnitude(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5

# Usage
v1 = Vector(3, 4)
v2 = Vector(1, 2)
v3 = v1 + v2  # __add__
v4 = 2 * v1   # __rmul__
len(v1)       # __len__ -> 2
v1[0]         # __getitem__ -> 3
v1(2)         # __call__ -> Vector(6, 8)
```

---

## Decorators

### Function Decorators
```python
import functools
import time
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')

# Basic decorator
def timer(func: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(func)  # Preserve function metadata
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)

# Decorator with arguments
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
            raise RuntimeError("Should not reach here")
        return wrapper
    return decorator

@retry(max_attempts=5, delay=0.5)
def unreliable_api_call():
    pass

# Stacking decorators (applied bottom-up)
@decorator1
@decorator2
@decorator3
def func():
    pass
# Equivalent to: decorator1(decorator2(decorator3(func)))

# functools.lru_cache (memoization)
@functools.lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# functools.cache (Python 3.9+, unbounded)
@functools.cache
def expensive_computation(x: int) -> int:
    return x ** 100
```

### Class Decorators
```python
# dataclass is a class decorator
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

# Custom class decorator
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        self.connection = "connected"

db1 = Database()
db2 = Database()
db1 is db2  # True
```

### Method Decorators
```python
class MyClass:
    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, v: int) -> None:
        self._value = v

    @classmethod
    def from_string(cls, s: str) -> "MyClass":
        return cls()

    @staticmethod
    def utility() -> None:
        pass
```

---

## Context Managers

### Using Context Managers
```python
# File handling (auto-closes)
with open("file.txt", "r") as f:
    content = f.read()

# Multiple context managers
with open("input.txt") as fin, open("output.txt", "w") as fout:
    fout.write(fin.read())

# Python 3.10+ parenthesized
with (
    open("input.txt") as fin,
    open("output.txt", "w") as fout,
):
    fout.write(fin.read())

# Database connection
with db.connect() as conn:
    conn.execute(query)

# Lock
from threading import Lock
lock = Lock()
with lock:
    # Thread-safe code
    pass
```

### Creating Context Managers
```python
from contextlib import contextmanager

# Class-based
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        print(f"Elapsed: {self.elapsed:.4f}s")
        return False  # Don't suppress exceptions

with Timer() as t:
    time.sleep(1)

# Generator-based (easier)
@contextmanager
def timer():
    start = time.perf_counter()
    try:
        yield  # Pause here, execute `with` block
    finally:
        elapsed = time.perf_counter() - start
        print(f"Elapsed: {elapsed:.4f}s")

with timer():
    time.sleep(1)

# Async context manager
class AsyncConnection:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

async with AsyncConnection() as conn:
    await conn.query()

# Generator-based async
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_timer():
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"Elapsed: {time.perf_counter() - start:.4f}s")
```

---

## Generators & Iterators

### Generators
```python
# Generator function (yields values lazily)
def count_up_to(n: int):
    i = 0
    while i < n:
        yield i
        i += 1

for num in count_up_to(5):
    print(num)  # 0, 1, 2, 3, 4

# Generator expression
squares = (x**2 for x in range(10))  # Note: parentheses, not brackets

# Infinite generator
def infinite_counter():
    n = 0
    while True:
        yield n
        n += 1

# Take first n items
from itertools import islice
first_10 = list(islice(infinite_counter(), 10))

# yield from (delegate to another generator)
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

list(flatten([[1, 2], [3, [4, 5]]]))  # [1, 2, 3, 4, 5]

# Generator with return value
def my_gen():
    yield 1
    yield 2
    return "done"

gen = my_gen()
next(gen)  # 1
next(gen)  # 2
# next(gen)  # StopIteration: done

# send() and throw()
def echo():
    while True:
        received = yield
        print(f"Got: {received}")

gen = echo()
next(gen)  # Prime the generator
gen.send("hello")  # Got: hello
```

### Itertools
```python
from itertools import (
    count, cycle, repeat,
    chain, compress, dropwhile, takewhile,
    groupby, islice, starmap,
    product, permutations, combinations
)

# Infinite iterators
count(10)        # 10, 11, 12, ...
cycle([1,2,3])   # 1, 2, 3, 1, 2, 3, ...
repeat(5, 3)     # 5, 5, 5

# Combining
chain([1,2], [3,4])  # 1, 2, 3, 4

# Filtering
list(takewhile(lambda x: x < 5, [1,3,5,2,4]))  # [1, 3]
list(dropwhile(lambda x: x < 5, [1,3,5,2,4]))  # [5, 2, 4]

# Grouping
data = [("a", 1), ("a", 2), ("b", 3)]
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, list(group))

# Combinatorics
list(product([1,2], [3,4]))      # [(1,3), (1,4), (2,3), (2,4)]
list(permutations([1,2,3], 2))   # All 2-length orderings
list(combinations([1,2,3], 2))   # All 2-length subsets
```

---

## Error Handling

### Try/Except
```python
try:
    result = risky_operation()
except ValueError as e:
    print(f"Value error: {e}")
except (TypeError, KeyError) as e:
    print(f"Type or key error: {e}")
except Exception as e:
    # Catch all other exceptions
    print(f"Unexpected error: {e}")
    raise  # Re-raise the exception
else:
    # Runs only if no exception
    print("Success!")
finally:
    # Always runs (cleanup)
    cleanup()

# Suppress exceptions
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove("file.txt")

# Re-raise with context
try:
    operation()
except ValueError as e:
    raise RuntimeError("Operation failed") from e
```

### Custom Exceptions
```python
class AppError(Exception):
    """Base exception for application."""
    pass

class ValidationError(AppError):
    """Validation failed."""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

class NotFoundError(AppError):
    """Resource not found."""
    def __init__(self, resource: str, id: str):
        self.resource = resource
        self.id = id
        super().__init__(f"{resource} with id {id} not found")

# Usage
raise ValidationError("email", "Invalid email format")
```

### Exception Groups (Python 3.11+)
```python
# Raise multiple exceptions
def process_items(items):
    errors = []
    for item in items:
        try:
            process(item)
        except Exception as e:
            errors.append(e)
    if errors:
        raise ExceptionGroup("Processing failed", errors)

# Handle exception groups
try:
    process_items(items)
except* ValueError as eg:
    print(f"Value errors: {eg.exceptions}")
except* TypeError as eg:
    print(f"Type errors: {eg.exceptions}")
```

---

## File I/O

### Reading Files
```python
# Read entire file
with open("file.txt", "r") as f:
    content = f.read()

# Read lines
with open("file.txt", "r") as f:
    lines = f.readlines()  # List with \n

with open("file.txt", "r") as f:
    lines = f.read().splitlines()  # List without \n

# Iterate lines (memory efficient)
with open("file.txt", "r") as f:
    for line in f:
        process(line.strip())

# Read with encoding
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()
```

### Writing Files
```python
# Write (overwrite)
with open("file.txt", "w") as f:
    f.write("Hello, World!\n")

# Append
with open("file.txt", "a") as f:
    f.write("New line\n")

# Write lines
lines = ["Line 1", "Line 2", "Line 3"]
with open("file.txt", "w") as f:
    f.writelines(line + "\n" for line in lines)

# Binary mode
with open("image.png", "rb") as f:
    data = f.read()

with open("output.png", "wb") as f:
    f.write(data)
```

### Pathlib (Modern Path Handling)
```python
from pathlib import Path

# Create path
p = Path("folder/subfolder/file.txt")
p = Path.home() / "Documents" / "file.txt"
p = Path.cwd() / "data"

# Path properties
p.name          # "file.txt"
p.stem          # "file"
p.suffix        # ".txt"
p.parent        # Path("folder/subfolder")
p.parts         # ("folder", "subfolder", "file.txt")
p.is_file()     # True/False
p.is_dir()      # True/False
p.exists()      # True/False

# Read/write
content = p.read_text()
p.write_text("Hello")
data = p.read_bytes()
p.write_bytes(data)

# Directory operations
p.mkdir(parents=True, exist_ok=True)
list(p.iterdir())  # List contents
list(p.glob("*.txt"))  # Glob pattern
list(p.rglob("*.py"))  # Recursive glob

# File operations
p.rename("new_name.txt")
p.unlink()  # Delete file
p.rmdir()   # Delete empty directory

import shutil
shutil.rmtree(p)  # Delete directory recursively
```

### JSON
```python
import json

# Parse JSON string
data = json.loads('{"name": "Claude", "age": 3}')

# Convert to JSON string
json_str = json.dumps({"name": "Claude"}, indent=2)

# Read JSON file
with open("data.json", "r") as f:
    data = json.load(f)

# Write JSON file
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

# Custom encoder
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

json.dumps(data, cls=CustomEncoder)
```

### YAML
```python
import yaml  # pip install pyyaml

# Read YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Write YAML
with open("config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)
```

### CSV
```python
import csv

# Read CSV
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        print(row)

# Read as dict
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["name"])

# Write CSV
with open("output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "age"])
    writer.writerows([["Alice", 30], ["Bob", 25]])
```

---

## Modules & Packages

### Imports
```python
# Import entire module
import os
os.path.join("a", "b")

# Import with alias
import numpy as np
np.array([1, 2, 3])

# Import specific items
from os.path import join, exists
join("a", "b")

# Import all (avoid in production)
from module import *

# Relative imports (within package)
from . import sibling_module
from .. import parent_module
from .sibling import function
```

### Package Structure
```
my_package/
├── __init__.py        # Package initializer
├── module1.py
├── module2.py
└── subpackage/
    ├── __init__.py
    └── module3.py
```

```python
# __init__.py - control what's exported
from .module1 import func1, Class1
from .module2 import func2

__all__ = ["func1", "Class1", "func2"]  # What * imports

# Version and metadata
__version__ = "1.0.0"
__author__ = "Your Name"
```

### Entry Points
```python
# my_module.py
def main():
    print("Hello from main!")

if __name__ == "__main__":
    main()

# Run: python -m my_module
```

### pyproject.toml (Modern)
```toml
[project]
name = "my-package"
version = "1.0.0"
description = "My awesome package"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "fastapi>=0.100",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
]

[project.scripts]
my-cli = "my_package.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Async/Await

### Basics
```python
import asyncio

# Async function (coroutine)
async def fetch_data(url: str) -> dict:
    await asyncio.sleep(1)  # Simulate I/O
    return {"data": "result"}

# Run async function
result = asyncio.run(fetch_data("http://example.com"))

# Await in async context
async def main():
    result = await fetch_data("http://example.com")
    print(result)

asyncio.run(main())
```

### Concurrent Execution
```python
import asyncio

async def fetch(url: str) -> str:
    await asyncio.sleep(1)
    return f"Data from {url}"

async def main():
    # Sequential (slow)
    result1 = await fetch("url1")
    result2 = await fetch("url2")

    # Concurrent (fast) - like Promise.all
    results = await asyncio.gather(
        fetch("url1"),
        fetch("url2"),
        fetch("url3"),
    )

    # With exception handling
    results = await asyncio.gather(
        fetch("url1"),
        fetch("url2"),
        return_exceptions=True,  # Don't raise, return exception
    )

    # TaskGroup (Python 3.11+)
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch("url1"))
        task2 = tg.create_task(fetch("url2"))
    # All tasks complete when exiting context

asyncio.run(main())
```

### Task Management
```python
async def main():
    # Create task (runs in background)
    task = asyncio.create_task(fetch("url"))

    # Do other work...

    # Wait for task
    result = await task

    # Cancel task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Task cancelled")

    # Timeout
    try:
        result = await asyncio.wait_for(fetch("url"), timeout=5.0)
    except asyncio.TimeoutError:
        print("Timeout!")

    # Wait for first to complete
    done, pending = await asyncio.wait(
        [fetch("url1"), fetch("url2")],
        return_when=asyncio.FIRST_COMPLETED,
    )
```

### Async Iterators
```python
# Async generator
async def async_range(n: int):
    for i in range(n):
        await asyncio.sleep(0.1)
        yield i

# Async for loop
async def main():
    async for num in async_range(5):
        print(num)

# Async comprehension
async def main():
    results = [x async for x in async_range(5)]
    filtered = [x async for x in async_range(5) if x % 2 == 0]
```

### Async Context Managers
```python
class AsyncDatabase:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

async def main():
    async with AsyncDatabase() as db:
        await db.query()

# With contextlib
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_connection():
    conn = await create_connection()
    try:
        yield conn
    finally:
        await conn.close()
```

### HTTP Client (httpx)
```python
import httpx

async def fetch_all():
    async with httpx.AsyncClient() as client:
        # Single request
        response = await client.get("https://api.example.com/data")
        data = response.json()

        # Parallel requests
        responses = await asyncio.gather(
            client.get("https://api.example.com/1"),
            client.get("https://api.example.com/2"),
        )

# With timeout and headers
async with httpx.AsyncClient(
    timeout=30.0,
    headers={"Authorization": "Bearer token"},
) as client:
    response = await client.post(
        "https://api.example.com/data",
        json={"key": "value"},
    )
```

### Anyio (Backend-Agnostic)
```python
import anyio

async def main():
    # Works with asyncio, trio, etc.
    await anyio.sleep(1)

    async with anyio.create_task_group() as tg:
        tg.start_soon(task1)
        tg.start_soon(task2)

anyio.run(main)
```

---

## Pydantic

### Basic Models
```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class User(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    email: str
    age: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)
    tags: list[str] = []

# Create from dict
user = User(id=1, name="Claude", email="claude@example.com")

# Access attributes
user.name
user.model_dump()  # To dict
user.model_dump_json()  # To JSON string

# Validate existing dict
data = {"id": 1, "name": "Claude", "email": "claude@example.com"}
user = User.model_validate(data)

# Parse JSON string
user = User.model_validate_json('{"id": 1, "name": "Claude", "email": "claude@example.com"}')
```

### Field Validation
```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Annotated

class User(BaseModel):
    # Field constraints
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    score: float = Field(..., gt=0, lt=100)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')

    # Annotated syntax (alternative)
    username: Annotated[str, Field(min_length=3, max_length=20)]

    # Field validator
    @field_validator('name')
    @classmethod
    def name_must_be_capitalized(cls, v: str) -> str:
        return v.title()

    # Validate multiple fields
    @field_validator('email', 'name')
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    # Model validator (access multiple fields)
    @model_validator(mode='after')
    def check_consistency(self) -> 'User':
        if self.age < 18 and 'adult' in self.tags:
            raise ValueError('Underage users cannot have adult tag')
        return self

# Before validator
@field_validator('age', mode='before')
@classmethod
def parse_age(cls, v):
    if isinstance(v, str):
        return int(v)
    return v
```

### Nested Models
```python
from pydantic import BaseModel
from typing import Optional

class Address(BaseModel):
    street: str
    city: str
    country: str = "USA"

class User(BaseModel):
    name: str
    address: Address
    friends: list["User"] = []

# Usage
user = User(
    name="Claude",
    address={"street": "123 Main St", "city": "SF"},
    friends=[{"name": "Alice", "address": {"street": "456 Oak", "city": "LA"}}]
)
```

### Config and Aliases
```python
from pydantic import BaseModel, ConfigDict, Field

class User(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,  # Validate on attribute assignment
        extra='forbid',  # Raise error on extra fields
        frozen=True,  # Immutable
        populate_by_name=True,  # Allow both alias and field name
    )

    # Alias for JSON keys
    user_id: int = Field(..., alias='userId')
    user_name: str = Field(..., alias='userName')

# Parse with aliases
data = {"userId": 1, "userName": "Claude"}
user = User.model_validate(data)

# Serialize with aliases
user.model_dump(by_alias=True)
```

### Custom Types
```python
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from typing import Any, Annotated

# Annotated validators
from pydantic import AfterValidator, BeforeValidator

def uppercase(v: str) -> str:
    return v.upper()

UpperStr = Annotated[str, AfterValidator(uppercase)]

class User(BaseModel):
    code: UpperStr  # Auto-uppercased

# Custom type with __get_pydantic_core_schema__
class CustomId:
    def __init__(self, value: str):
        self.value = value

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
        )
```

### Pydantic Settings
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_prefix='APP_',
    )

    database_url: str
    api_key: str
    debug: bool = False
    port: int = 8000

# Reads from environment variables: APP_DATABASE_URL, APP_API_KEY, etc.
settings = Settings()
```

---

## FastAPI

### Basic App
```python
from fastapi import FastAPI, HTTPException, Query, Path, Body, Depends
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="My API", version="1.0.0")

class Item(BaseModel):
    name: str
    price: float
    description: Optional[str] = None

# GET endpoint
@app.get("/")
async def root():
    return {"message": "Hello, World!"}

# Path parameters
@app.get("/items/{item_id}")
async def get_item(
    item_id: int = Path(..., ge=1, description="Item ID"),
):
    return {"item_id": item_id}

# Query parameters
@app.get("/items")
async def list_items(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    q: Optional[str] = Query(None, min_length=3),
):
    return {"skip": skip, "limit": limit, "q": q}

# POST with body
@app.post("/items", status_code=201)
async def create_item(item: Item):
    return {"item": item, "id": 1}

# PUT
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, "item": item}

# DELETE
@app.delete("/items/{item_id}", status_code=204)
async def delete_item(item_id: int):
    return None
```

### Dependency Injection
```python
from fastapi import Depends, HTTPException, Header

# Simple dependency
async def get_db():
    db = Database()
    try:
        yield db
    finally:
        await db.close()

@app.get("/users")
async def get_users(db = Depends(get_db)):
    return await db.fetch_all("users")

# Dependency with parameters
def pagination(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
) -> dict:
    return {"skip": skip, "limit": limit}

@app.get("/items")
async def list_items(pagination: dict = Depends(pagination)):
    return pagination

# Auth dependency
async def get_current_user(
    authorization: str = Header(...),
) -> User:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid token")
    token = authorization[7:]
    user = await verify_token(token)
    if not user:
        raise HTTPException(401, "Invalid token")
    return user

@app.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    return user

# Class-based dependency
class DatabaseSession:
    def __init__(self, db_url: str = Depends(get_db_url)):
        self.db_url = db_url

    async def __call__(self):
        return await create_session(self.db_url)
```

### Response Models
```python
from fastapi import FastAPI
from pydantic import BaseModel

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str  # No password!

@app.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate):
    # password is stripped from response
    return {"id": 1, **user.model_dump()}

# Multiple response types
from typing import Union

@app.get("/items/{item_id}", response_model=Union[FullItem, PartialItem])
async def get_item(item_id: int, full: bool = False):
    if full:
        return FullItem(...)
    return PartialItem(...)
```

### Error Handling
```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Raise HTTP exception
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    item = await find_item(item_id)
    if not item:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "Item not found"},
        )
    return item

# Custom exception
class ItemNotFoundError(Exception):
    def __init__(self, item_id: int):
        self.item_id = item_id

@app.exception_handler(ItemNotFoundError)
async def item_not_found_handler(request: Request, exc: ItemNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"message": f"Item {exc.item_id} not found"},
    )

# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )
```

### Background Tasks
```python
from fastapi import BackgroundTasks

async def send_email(email: str, message: str):
    # Long-running task
    await asyncio.sleep(5)
    print(f"Email sent to {email}")

@app.post("/send-notification")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks,
):
    background_tasks.add_task(send_email, email, "Hello!")
    return {"message": "Notification sent in background"}
```

### Middleware
```python
from fastapi.middleware.cors import CORSMiddleware
import time

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### Routers
```python
# routers/users.py
from fastapi import APIRouter

router = APIRouter(
    prefix="/users",
    tags=["users"],
)

@router.get("/")
async def list_users():
    return []

@router.get("/{user_id}")
async def get_user(user_id: int):
    return {"id": user_id}

# main.py
from fastapi import FastAPI
from routers import users

app = FastAPI()
app.include_router(users.router)
```

### Lifespan Events
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await database.connect()
    yield
    # Shutdown
    await database.disconnect()

app = FastAPI(lifespan=lifespan)
```

---

## Testing with Pytest

### Basic Tests
```python
# test_example.py
import pytest

def test_addition():
    assert 1 + 1 == 2

def test_string():
    assert "hello".upper() == "HELLO"

def test_list():
    lst = [1, 2, 3]
    assert len(lst) == 3
    assert 2 in lst

# Expected failure
def test_exception():
    with pytest.raises(ValueError):
        int("not a number")

def test_exception_message():
    with pytest.raises(ValueError, match="invalid literal"):
        int("not a number")
```

### Fixtures
```python
import pytest

@pytest.fixture
def sample_data():
    return {"name": "test", "value": 42}

def test_with_fixture(sample_data):
    assert sample_data["name"] == "test"

# Setup and teardown
@pytest.fixture
def database():
    db = Database()
    db.connect()
    yield db  # Provide to test
    db.disconnect()  # Teardown

# Scope (function, class, module, session)
@pytest.fixture(scope="module")
def shared_resource():
    return expensive_setup()

# Parameterized fixture
@pytest.fixture(params=[1, 2, 3])
def number(request):
    return request.param

def test_with_param(number):
    assert number > 0

# Auto-use fixture
@pytest.fixture(autouse=True)
def setup_logging():
    logging.basicConfig(level=logging.DEBUG)
```

### Async Tests
```python
import pytest
import asyncio

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.mark.asyncio
async def test_async_function():
    result = await async_fetch("url")
    assert result == expected

# With pytest-asyncio
# pip install pytest-asyncio

# pytest.ini or pyproject.toml:
# [tool.pytest.ini_options]
# asyncio_mode = "auto"

async def test_auto_async():
    await asyncio.sleep(0.1)
    assert True
```

### FastAPI Testing
```python
from fastapi.testclient import TestClient
from httpx import AsyncClient
import pytest

from main import app

# Sync client
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}

def test_create_item():
    response = client.post(
        "/items",
        json={"name": "Test", "price": 10.0},
    )
    assert response.status_code == 201

# Async client
@pytest.mark.asyncio
async def test_async_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
        assert response.status_code == 200

# Override dependencies
def override_get_db():
    return MockDatabase()

app.dependency_overrides[get_db] = override_get_db

def test_with_mock_db():
    response = client.get("/items")
    assert response.status_code == 200
```

### Mocking
```python
from unittest.mock import Mock, patch, AsyncMock

# Mock object
mock = Mock()
mock.return_value = 42
mock.method.return_value = "result"

# Patch
@patch("module.function")
def test_with_patch(mock_func):
    mock_func.return_value = "mocked"
    result = function_that_calls_module_function()
    assert result == "mocked"
    mock_func.assert_called_once()

# Context manager
def test_with_context():
    with patch("module.function") as mock_func:
        mock_func.return_value = "mocked"
        result = call_function()
        assert result == "mocked"

# Async mock
@pytest.mark.asyncio
async def test_async_mock():
    mock = AsyncMock(return_value="result")
    result = await mock()
    assert result == "result"

# patch.object
@patch.object(MyClass, "method", return_value="mocked")
def test_method(mock_method):
    obj = MyClass()
    assert obj.method() == "mocked"
```

### Markers and Configuration
```python
# Markers
@pytest.mark.slow
def test_slow_operation():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_skip():
    pass

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
def test_conditional():
    pass

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert input * 2 == expected

# pyproject.toml
# [tool.pytest.ini_options]
# markers = [
#     "slow: marks tests as slow",
#     "integration: integration tests",
# ]
# filterwarnings = ["ignore::DeprecationWarning"]
```

---

## Agent SDK Patterns

### Claude Agent SDK
```python
import anyio
from claude_agent_sdk import query, ClaudeSDKClient
from claude_agent_sdk.tools import Tool
from pydantic import BaseModel

# Simple query
async def simple_agent():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

# Custom tools
class CalculatorInput(BaseModel):
    expression: str

class CalculatorOutput(BaseModel):
    result: float

@Tool(
    name="calculator",
    description="Evaluate math expressions"
)
async def calculator(input: CalculatorInput) -> CalculatorOutput:
    result = eval(input.expression)  # In production, use a safe parser
    return CalculatorOutput(result=result)

# Client with tools
async def agent_with_tools():
    client = ClaudeSDKClient(
        tools=[calculator],
        model="claude-sonnet-4-20250514",
    )

    async for msg in client.query("Calculate 15 * 23"):
        print(msg)

# Hooks for validation
async def pre_tool_hook(tool_name: str, input: dict) -> dict:
    print(f"About to call {tool_name} with {input}")
    return input  # Can modify input

async def post_tool_hook(tool_name: str, output: dict) -> dict:
    print(f"{tool_name} returned {output}")
    return output  # Can modify output

client = ClaudeSDKClient(
    tools=[calculator],
    pre_tool_hook=pre_tool_hook,
    post_tool_hook=post_tool_hook,
)

anyio.run(agent_with_tools)
```

### Pydantic AI
```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

# Structured output
class Analysis(BaseModel):
    sentiment: str
    confidence: float
    key_points: list[str]

agent = Agent(
    'claude-3-5-sonnet-20241022',
    result_type=Analysis,
    system_prompt="Analyze text and return structured analysis."
)

async def analyze(text: str) -> Analysis:
    result = await agent.run(f"Analyze: {text}")
    return result.data  # Type-safe Analysis

# With tools
@agent.tool
async def search_database(query: str) -> list[dict]:
    """Search the database for relevant records."""
    return [{"id": 1, "name": "Result"}]

# With dependencies (for testing)
from dataclasses import dataclass

@dataclass
class Deps:
    api_key: str
    database: Database

agent = Agent(
    'claude-3-5-sonnet-20241022',
    deps_type=Deps,
)

@agent.tool
async def query_api(ctx, endpoint: str) -> dict:
    # Access dependencies via ctx.deps
    return await fetch(endpoint, key=ctx.deps.api_key)

# Run with dependencies
async def main():
    deps = Deps(api_key="...", database=db)
    result = await agent.run("Query the API", deps=deps)

# Multi-agent delegation
research_agent = Agent('claude-3-5-sonnet', system_prompt="Research")
writer_agent = Agent('claude-3-5-sonnet', system_prompt="Write")

@writer_agent.tool
async def research(topic: str) -> str:
    result = await research_agent.run(f"Research: {topic}")
    return result.data
```

### Google ADK
```python
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.tools import tool

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def analyze_data(data: str) -> dict:
    """Analyze the provided data."""
    return {"analysis": "complete", "data": data}

# Single agent
researcher = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction="You research topics thoroughly and accurately.",
    tools=[search_web]
)

# Sequential pipeline
summarizer = Agent(
    name="summarizer",
    model="gemini-2.0-flash",
    instruction="You summarize research into key points."
)

pipeline = SequentialAgent(
    name="research_pipeline",
    sub_agents=[researcher, summarizer]
)

# Parallel execution
analyst1 = Agent(name="analyst1", model="gemini-2.0-flash")
analyst2 = Agent(name="analyst2", model="gemini-2.0-flash")

parallel = ParallelAgent(
    name="parallel_analysis",
    sub_agents=[analyst1, analyst2]
)

# Human-in-the-loop
from google.adk.tools import ToolConfirmation

@tool(confirmation=ToolConfirmation.REQUIRED)
def delete_file(path: str) -> str:
    """Delete a file. Requires confirmation."""
    import os
    os.remove(path)
    return f"Deleted: {path}"
```

---

## Common Patterns

### Singleton
```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Or with decorator
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    pass
```

### Factory
```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"

def animal_factory(animal_type: str) -> Animal:
    animals = {
        "dog": Dog,
        "cat": Cat,
    }
    return animals[animal_type]()
```

### Registry
```python
from typing import Dict, Type, Callable

# Class registry
class HandlerRegistry:
    _handlers: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(handler_cls):
            cls._handlers[name] = handler_cls
            return handler_cls
        return decorator

    @classmethod
    def get(cls, name: str):
        return cls._handlers[name]

@HandlerRegistry.register("email")
class EmailHandler:
    pass

@HandlerRegistry.register("sms")
class SMSHandler:
    pass

handler = HandlerRegistry.get("email")()
```

### Retry with Backoff
```python
import asyncio
from functools import wraps

def retry(max_attempts: int = 3, backoff: float = 1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait = backoff * (2 ** attempt)
                    await asyncio.sleep(wait)
        return wrapper
    return decorator

@retry(max_attempts=5, backoff=0.5)
async def flaky_api_call():
    pass
```

### Rate Limiter
```python
import asyncio
import time
from collections import deque

class RateLimiter:
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.timestamps = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()

            # Remove old timestamps
            while self.timestamps and now - self.timestamps[0] > self.period:
                self.timestamps.popleft()

            if len(self.timestamps) >= self.calls:
                sleep_time = self.period - (now - self.timestamps[0])
                await asyncio.sleep(sleep_time)
                return await self.acquire()

            self.timestamps.append(now)

limiter = RateLimiter(calls=10, period=1.0)

async def limited_call():
    await limiter.acquire()
    return await api_call()
```

### Circuit Breaker
```python
import time
from enum import Enum

class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = State.CLOSED
        self.last_failure_time = 0

    async def call(self, func, *args, **kwargs):
        if self.state == State.OPEN:
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = State.HALF_OPEN
            else:
                raise Exception("Circuit is open")

        try:
            result = await func(*args, **kwargs)
            if self.state == State.HALF_OPEN:
                self.state = State.CLOSED
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = State.OPEN
            raise
```

---

## Standard Library Gems

### functools
```python
from functools import (
    lru_cache, cache, cached_property,
    partial, reduce, wraps, singledispatch
)

# Memoization
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Unbounded cache (Python 3.9+)
@cache
def expensive(x):
    return x ** 100

# Cached property
class Circle:
    def __init__(self, radius):
        self.radius = radius

    @cached_property
    def area(self):
        return 3.14159 * self.radius ** 2

# Single dispatch (function overloading)
@singledispatch
def process(arg):
    print(f"Default: {arg}")

@process.register(int)
def _(arg):
    print(f"Integer: {arg}")

@process.register(list)
def _(arg):
    print(f"List with {len(arg)} items")
```

### dataclasses
```python
from dataclasses import dataclass, field, asdict, astuple, replace

@dataclass
class Point:
    x: float
    y: float
    label: str = ""

    def distance_from_origin(self):
        return (self.x**2 + self.y**2) ** 0.5

p = Point(3, 4)
asdict(p)  # {'x': 3, 'y': 4, 'label': ''}
astuple(p)  # (3, 4, '')
replace(p, x=5)  # Point(5, 4, '')
```

### logging
```python
import logging

# Basic setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.exception("Exception with traceback")

# Structured logging
logger.info("User logged in", extra={"user_id": 123, "ip": "1.2.3.4"})
```

### datetime
```python
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+

# Current time
now = datetime.now()
utc_now = datetime.now(ZoneInfo("UTC"))

# Create datetime
dt = datetime(2024, 1, 15, 10, 30, 0)

# Parse string
dt = datetime.fromisoformat("2024-01-15T10:30:00")
dt = datetime.strptime("2024-01-15", "%Y-%m-%d")

# Format
dt.isoformat()  # '2024-01-15T10:30:00'
dt.strftime("%B %d, %Y")  # 'January 15, 2024'

# Arithmetic
tomorrow = now + timedelta(days=1)
diff = datetime(2024, 12, 31) - now

# Timezones
utc = ZoneInfo("UTC")
pacific = ZoneInfo("America/Los_Angeles")
dt_pacific = now.astimezone(pacific)
```

### uuid
```python
import uuid

# Generate UUIDs
uuid.uuid4()  # Random UUID
uuid.uuid1()  # Based on host ID and time

# From string
u = uuid.UUID('12345678-1234-5678-1234-567812345678')
str(u)  # Back to string
```

### secrets
```python
import secrets

# Secure random
secrets.token_bytes(32)  # Random bytes
secrets.token_hex(32)    # Hex string
secrets.token_urlsafe(32)  # URL-safe string

# Secure comparison (timing attack safe)
secrets.compare_digest(a, b)
```

### hashlib
```python
import hashlib

# Hash data
data = b"Hello, World!"
hashlib.sha256(data).hexdigest()
hashlib.md5(data).hexdigest()

# Hash file
def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
```

### os and sys
```python
import os
import sys

# Environment variables
os.environ.get("HOME")
os.environ.setdefault("DEBUG", "false")

# Paths
os.getcwd()
os.path.join("a", "b", "c")
os.path.exists("file.txt")
os.path.dirname(__file__)

# System info
sys.version
sys.platform
sys.argv  # Command line arguments
sys.exit(1)  # Exit with code
```

### re (Regular Expressions)
```python
import re

# Match
if re.match(r'^\d+$', '123'):
    print("All digits")

# Search (find anywhere)
match = re.search(r'\d+', 'abc123def')
if match:
    print(match.group())  # '123'

# Find all
numbers = re.findall(r'\d+', 'a1b2c3')  # ['1', '2', '3']

# Replace
result = re.sub(r'\s+', ' ', 'too   many   spaces')

# Compile for reuse
pattern = re.compile(r'(\w+)@(\w+)\.(\w+)')
match = pattern.search('user@example.com')
if match:
    match.groups()  # ('user', 'example', 'com')

# Named groups
pattern = re.compile(r'(?P<user>\w+)@(?P<domain>\w+\.w+)')
match = pattern.search('user@example.com')
match.group('user')  # 'user'
```

---

## Quick Reference Cards

### String Methods
| Method | Description |
|--------|-------------|
| `s.strip()` | Remove whitespace |
| `s.split(',')` | Split by delimiter |
| `','.join(lst)` | Join list with delimiter |
| `s.replace(a, b)` | Replace substring |
| `s.find(sub)` | Find index (-1 if not found) |
| `s.startswith(pre)` | Check prefix |
| `s.endswith(suf)` | Check suffix |
| `s.upper()` / `s.lower()` | Case conversion |
| `s.title()` | Title Case |
| `s.isdigit()` / `s.isalpha()` | Check content |

### List Methods
| Method | Description |
|--------|-------------|
| `lst.append(x)` | Add to end |
| `lst.extend(iter)` | Add multiple |
| `lst.insert(i, x)` | Insert at index |
| `lst.remove(x)` | Remove first occurrence |
| `lst.pop()` | Remove and return last |
| `lst.pop(i)` | Remove and return at index |
| `lst.sort()` | Sort in place |
| `sorted(lst)` | Return sorted copy |
| `lst.reverse()` | Reverse in place |
| `lst.index(x)` | Find index |
| `lst.count(x)` | Count occurrences |

### Dict Methods
| Method | Description |
|--------|-------------|
| `d.get(k, default)` | Get with default |
| `d.keys()` / `d.values()` / `d.items()` | Views |
| `d.update(other)` | Merge dicts |
| `d.pop(k, default)` | Remove and return |
| `d.setdefault(k, v)` | Set if missing |
| `d.fromkeys(keys, v)` | Create from keys |
| `k in d` | Check key exists |

### Common Built-ins
| Function | Description |
|----------|-------------|
| `len(x)` | Length |
| `range(start, stop, step)` | Integer sequence |
| `enumerate(iter)` | Index + value pairs |
| `zip(a, b)` | Parallel iteration |
| `map(fn, iter)` | Apply function |
| `filter(fn, iter)` | Filter by predicate |
| `sorted(iter, key=fn)` | Sorted copy |
| `reversed(iter)` | Reversed iterator |
| `any(iter)` / `all(iter)` | Boolean reduce |
| `min(iter)` / `max(iter)` | Extremes |
| `sum(iter)` | Sum of numbers |
| `isinstance(obj, type)` | Type check |
| `hasattr(obj, name)` | Attribute check |
| `getattr(obj, name, default)` | Get attribute |

---

## Related Notes

- [python](python.md) - Learning curriculum
- [full-stack-development](full-stack-development.md) - Tech stack
- [fastAPI](fastAPI.md) - API framework details
- [pytest](pytest.md) - Testing details

