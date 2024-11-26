import csv
import random
import copy
import re
import math

from tabulate import tabulate


class Auditorium:
    """
    Represents an auditorium with a unique ID and seating capacity.

    Attributes:
        id (str): Unique identifier for the auditorium.
        capacity (int): The seating capacity of the auditorium.
    """

    def __init__(self, auditorium_id, capacity):
        self.id = auditorium_id  # Set the unique ID for the auditorium.
        self.capacity = int(capacity)  # Ensure the capacity is stored as an integer.


class Group:
    """
    Represents a student group, including possible subgroups.

    Attributes:
        number (str): Unique identifier for the group.
        size (int): The number of students in the group.
        subgroups (list[str]): List of subgroup identifiers if applicable.
    """

    def __init__(self, group_number, student_amount, subgroups):
        self.number = group_number  # Assign the group number.
        self.size = int(
            student_amount
        )  # Ensure the student count is stored as an integer.
        # Parse and split subgroups if provided; otherwise, initialize as an empty list.
        self.subgroups = subgroups.strip('"').split(";") if subgroups else []


class Lecturer:
    """
    Represents a lecturer with teaching capabilities and workload constraints.

    Attributes:
        id (str): Unique identifier for the lecturer.
        name (str): Name of the lecturer.
        subjects_can_teach (list[str]): List of subjects the lecturer can teach.
        types_can_teach (list[str]): List of teaching types (e.g., Lecture, Practice).
        max_hours_per_week (int): Maximum hours the lecturer can teach per week.
    """

    def __init__(
        self, lecturer_id, name, subjects_can_teach, types_can_teach, max_hours_per_week
    ):
        self.id = lecturer_id  # Set the unique ID for the lecturer.
        self.name = name  # Assign the lecturer's name.

        # Parse the list of subjects the lecturer can teach, splitting by semicolon or comma.
        self.subjects_can_teach = (
            [s.strip() for s in re.split(";|,", subjects_can_teach)]
            if subjects_can_teach
            else []
        )

        # Parse the list of teaching types (e.g., Lecture, Practice).
        self.types_can_teach = (
            [t.strip() for t in re.split(";|,", types_can_teach)]
            if types_can_teach
            else []
        )

        # Ensure the maximum hours per week is stored as an integer.
        self.max_hours_per_week = int(max_hours_per_week)


class Subject:
    """
    Represents a subject with associated group, lecture, and practical requirements.

    Attributes:
        id (str): Unique identifier for the subject.
        name (str): Name of the subject.
        group_id (str): Identifier for the associated student group.
        num_lectures (int): Number of lectures required for this subject.
        num_practicals (int): Number of practicals required for this subject.
        requires_subgroups (bool): Whether practicals require subgroups.
        week_type (str): The type of week ('both', 'even', or 'odd') the subject is scheduled for.
    """

    def __init__(
        self,
        subject_id,
        name,
        group_id,
        num_lectures,
        num_practicals,
        requires_subgroups,
        week_type,
    ):
        self.id = subject_id  # Set the unique ID for the subject.
        self.name = name  # Assign the name of the subject.
        self.group_id = group_id  # Link the subject to a specific group.
        self.num_lectures = int(num_lectures)  # Ensure lecture count is an integer.
        self.num_practicals = int(
            num_practicals
        )  # Ensure practical count is an integer.
        # Determine if practicals require subgroups based on input.
        self.requires_subgroups = True if requires_subgroups.lower() == "yes" else False
        self.week_type = week_type.lower()  # Normalize week type to lowercase.


class Lesson:
    """
    Represents a lesson, including subject, type, group, and scheduling details.

    Attributes:
        subject (Subject): The subject being taught in the lesson.
        type (str): Type of lesson ('Lecture' or 'Practice').
        group (Group): The student group associated with the lesson.
        subgroup (str, optional): Subgroup identifier if applicable.
        time_slot (tuple, optional): The scheduled time slot for the lesson (day, period).
        auditorium (Auditorium, optional): The assigned auditorium for the lesson.
        lecturer (Lecturer, optional): The assigned lecturer for the lesson.
    """

    def __init__(self, subject, lesson_type, group, subgroup=None):
        self.subject = subject  # Assign the subject for the lesson.
        self.type = lesson_type  # Specify the type of lesson ('Lecture' or 'Practice').
        self.group = group  # Link the lesson to a student group.
        self.subgroup = subgroup  # Optional: specify a subgroup for practical lessons.
        self.time_slot = None  # Initialize the time slot as None.
        self.auditorium = None  # Initialize the auditorium as None.
        self.lecturer = None  # Initialize the lecturer as None.


class TwoWeeksSchedule:
    """
    Represents a schedule for both even and odd weeks, including the fitness calculation.

    Attributes:
        even_timetable (dict): TwoWeeksSchedule for even weeks, with time slots as keys and lists of lessons as values.
        odd_timetable (dict): TwoWeeksSchedule for odd weeks, similar structure as even_timetable.
        fitness (float): Fitness value of the schedule, calculated based on penalties for various constraints.
    """

    def __init__(self):
        # Initialize empty timetables for even and odd weeks.
        # Each time_slot (tuple of day and period) maps to a list of lessons scheduled for that time.
        self.even_timetable = {time_slot: [] for time_slot in TIME_SLOTS}
        self.odd_timetable = {time_slot: [] for time_slot in TIME_SLOTS}
        self.fitness = None  # Fitness will be calculated based on penalties.

    def calculate_fitness(self):
        """
        Calculates the fitness of the schedule by summing penalties for various constraints:
        - Gaps in group and lecturer schedules.
        - Exceeding or not meeting required hours for subjects.
        """
        penalty = 0
        # Add penalties for even-week schedule constraints.
        penalty += self._calculate_fitness_for_week(self.even_timetable)
        # Add penalties for odd-week schedule constraints.
        penalty += self._calculate_fitness_for_week(self.odd_timetable)
        # Add penalties for soft constraints related to subjects.
        penalty += self._calculate_soft_constraints()
        # Ensure penalty is non-negative and calculate fitness (higher is better).
        if penalty < 0:
            penalty = 0
        self.fitness = 1 / (1 + penalty)

    def _calculate_fitness_for_week(self, timetable):
        """
        Calculates penalties for a single week's timetable.

        Args:
            timetable (dict): Timetable for either even or odd week.

        Returns:
            int: Total penalty for the timetable.
        """
        penalty = 0
        # Minimize gaps in group schedules (soft constraint).
        for group in groups:
            subgroups = group.subgroups if group.subgroups else [None]
            for subgroup in subgroups:
                schedule_list = (
                    []
                )  # Store all scheduled time slots for the group or subgroup.
                for time_slot, lessons in timetable.items():
                    for lesson in lessons:
                        if (
                            lesson.group.number == group.number
                            and lesson.subgroup == subgroup
                        ):
                            schedule_list.append(time_slot)
                # Sort time slots by day and period.
                schedule_sorted = sorted(
                    schedule_list, key=lambda x: (DAYS.index(x[0]), int(x[1]))
                )
                # Calculate gaps between consecutive time slots.
                for i in range(len(schedule_sorted) - 1):
                    day1, period1 = schedule_sorted[i]
                    day2, period2 = schedule_sorted[i + 1]
                    if day1 == day2:  # Gaps are calculated within the same day.
                        gaps = int(period2) - int(period1) - 1
                        if gaps > 0:
                            penalty += gaps

        # Minimize gaps in lecturer schedules (soft constraint).
        for lecturer in lecturers:
            schedule_list = []  # Store all scheduled time slots for the lecturer.
            for time_slot, lessons in timetable.items():
                for lesson in lessons:
                    if lesson.lecturer and lesson.lecturer.id == lecturer.id:
                        schedule_list.append(time_slot)
            # Sort time slots by day and period.
            schedule_sorted = sorted(
                schedule_list, key=lambda x: (DAYS.index(x[0]), int(x[1]))
            )
            # Calculate gaps between consecutive time slots.
            for i in range(len(schedule_sorted) - 1):
                day1, period1 = schedule_sorted[i]
                day2, period2 = schedule_sorted[i + 1]
                if day1 == day2:  # Gaps are calculated within the same day.
                    gaps = int(period2) - int(period1) - 1
                    if gaps > 0:
                        penalty += gaps

            # Penalize exceeding the lecturer's maximum allowed hours per week.
            hours_assigned = len(
                schedule_list
            )  # Total number of lessons assigned to this lecturer.
            max_hours = lecturer.max_hours_per_week
            if hours_assigned > max_hours:
                penalty += (
                    hours_assigned - max_hours
                ) * 2  # Higher penalty for exceeding hours.

        return penalty

    def _calculate_soft_constraints(self):
        """
        Calculates penalties for not meeting or exceeding the required number of hours per subject.

        Returns:
            int: Total penalty for soft constraint violations.
        """
        penalty = 0
        for subject in subjects:
            # Find the group associated with the subject.
            group = next((g for g in groups if g.number == subject.group_id), None)
            if not group:
                continue

            subgroups = group.subgroups if subject.requires_subgroups else [None]
            scheduled_lectures = 0
            scheduled_practicals = {
                s: 0 for s in subgroups
            }  # Count of practicals per subgroup.
            required_lectures = subject.num_lectures
            required_practicals = subject.num_practicals

            # Count scheduled lectures and practicals for both weeks.
            for subgroup in subgroups:
                for timetable in [self.even_timetable, self.odd_timetable]:
                    for time_slot, lessons in timetable.items():
                        for lesson in lessons:
                            if (
                                lesson.type == "Лекція"
                                and lesson.subject.id == subject.id
                                and lesson.group.number == group.number
                            ):
                                scheduled_lectures += 1
                            elif (
                                lesson.type == "Практика"
                                and lesson.subject.id == subject.id
                                and lesson.group.number == group.number
                                and lesson.subgroup == subgroup
                            ):
                                scheduled_practicals[subgroup] += 1

            # Calculate penalties for mismatched lecture counts.
            diff_lectures = scheduled_lectures - required_lectures
            penalty += (
                abs(diff_lectures) * 2
            )  # Penalize deviation from required lecture hours.

            # Calculate penalties for mismatched practical counts.
            diff_practicals = [
                abs(practical - required_practicals)
                for _, practical in scheduled_practicals.items()
            ]
            penalty += (
                sum(diff_practicals) * 2
            )  # Penalize deviation from required practical hours.

        return penalty


def get_possible_lecturers(lesson):
    """
    Finds all lecturers capable of teaching a given lesson.

    Args:
        lesson (Lesson): The lesson for which to find possible lecturers.

    Returns:
        list[Lecturer]: List of lecturers who can teach the given lesson.
    """
    # Matching lecturers by subject.id and lesson type (hard constraint)
    possible = [
        lecturer
        for lecturer in lecturers
        if lesson.subject.id in lecturer.subjects_can_teach
        and lesson.type in lecturer.types_can_teach
    ]
    if not possible:
        # Print a warning if no lecturer can teach the lesson
        print(
            f"No lecturer available for {lesson.subject.name} ({lesson.type}) "
            f"with subject ID {lesson.subject.id}."
        )
    return possible


# Functions to read CSV files
def read_auditoriums_csv(filename):
    """
    Reads auditorium data from a CSV file.

    Args:
        filename (str): The path to the CSV file containing auditorium data.

    Returns:
        list[Auditorium]: List of Auditorium objects.
    """
    auditoriums = []
    with open(filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            auditoriums.append(Auditorium(row["auditoriumID"], row["capacity"]))
    return auditoriums


def read_groups_csv(filename):
    """
    Reads group data from a CSV file.

    Args:
        filename (str): The path to the CSV file containing group data.

    Returns:
        list[Group]: List of Group objects.
    """
    groups = []
    with open(filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            groups.append(
                Group(row["groupNumber"], row["studentAmount"], row["subgroups"])
            )
    return groups


def read_lecturers_csv(filename):
    """
    Reads lecturer data from a CSV file.

    Args:
        filename (str): The path to the CSV file containing lecturer data.

    Returns:
        list[Lecturer]: List of Lecturer objects.
    """
    lecturers = []
    with open(filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            lecturers.append(
                Lecturer(
                    row["lecturerID"],
                    row["lecturerName"],
                    row["subjectsCanTeach"],
                    row["typesCanTeach"],
                    row["maxHoursPerWeek"],
                )
            )
    return lecturers


def read_subjects_csv(filename):
    """
    Reads subject data from a CSV file.

    Args:
        filename (str): The path to the CSV file containing subject data.

    Returns:
        list[Subject]: List of Subject objects.
    """
    subjects = []
    with open(filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            subjects.append(
                Subject(
                    row["id"],
                    row["name"],
                    row["groupID"],
                    row["numLectures"],
                    row["numPracticals"],
                    row["requiresSubgroups"],
                    row["weekType"],
                )
            )
    return subjects


# Loading data
auditoriums = read_auditoriums_csv("assets1/auditoriums.csv")
groups = read_groups_csv("assets1/groups.csv")
lecturers = read_lecturers_csv("assets1/lecturers.csv")
subjects = read_subjects_csv("assets1/subjects.csv")

# Checking that each subject has at least one lecturer
subject_ids = set(subject.id for subject in subjects)
lecturer_subjects = set()
for lecturer in lecturers:
    lecturer_subjects.update(lecturer.subjects_can_teach)

# Identify missing subjects with no assigned lecturers
missing_subjects = subject_ids - lecturer_subjects
if missing_subjects:
    print(
        f"Warning: No lecturers available for the following subjects: {', '.join(missing_subjects)}"
    )

# Defining time slots: 5 days, 4 periods per day
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
PERIODS = ["1", "2", "3", "4"]  # Periods per day
TIME_SLOTS = [(day, period) for day in DAYS for period in PERIODS]


def does_lesson_conflict(lesson, time_slot, timetable):
    """
    Checks if a lesson conflicts with existing lessons in the given time slot.

    Args:
        lesson (Lesson): The lesson to check for conflicts.
        time_slot (tuple): The time slot being checked (day, period).
        timetable (dict): The timetable containing scheduled lessons.

    Returns:
        bool: True if there is a conflict, False otherwise.
    """
    for existing_lesson in timetable[time_slot]:
        # Check for lecturer conflict (hard constraint)
        if (
            lesson.lecturer
            and existing_lesson.lecturer
            and existing_lesson.lecturer.id == lesson.lecturer.id
        ):
            return True
        # Check for auditorium conflict (hard constraint)
        if (
            lesson.auditorium
            and existing_lesson.auditorium
            and existing_lesson.auditorium.id == lesson.auditorium.id
        ):
            return True
        # Check for group and subgroup conflict (hard constraint)
        if lesson.group.number == existing_lesson.group.number:
            if lesson.subgroup == existing_lesson.subgroup:
                return True
            # If one of the lessons lacks subgroups, it conflicts with all subgroups
            if not lesson.subgroup or not existing_lesson.subgroup:
                return True
    return False


# Genetic algorithm settings
POPULATION_SIZE = 100  # Number of schedules in each generation
GENERATIONS = 200  # Maximum number of generations for optimization


def create_initial_population():
    """
    Creates the initial population of schedules for the genetic algorithm.

    Each schedule is initialized with randomly assigned lessons while ensuring
    constraints like lecturer availability and auditorium capacity are met.

    Returns:
        list[TwoWeeksSchedule]: A list of TwoWeeksSchedule objects representing the initial population.
    """
    population = []
    for _ in range(POPULATION_SIZE):
        schedule = TwoWeeksSchedule()
        lessons_to_schedule = []

        # Generate all lessons to be scheduled based on subjects
        for subject in subjects:
            group = next((g for g in groups if g.number == subject.group_id), None)
            if not group:
                continue

            # Add required number of lecture lessons
            for _ in range(subject.num_lectures):
                lessons_to_schedule.append(Lesson(subject, "Лекція", group))

            # Add required number of practical lessons
            if subject.requires_subgroups and group.subgroups:
                # Distribute practical lessons across subgroups
                num_practicals_per_subgroup = math.ceil(
                    subject.num_practicals / len(group.subgroups)
                )
                for subgroup in group.subgroups:
                    for _ in range(num_practicals_per_subgroup):
                        lessons_to_schedule.append(
                            Lesson(subject, "Практика", group, subgroup)
                        )
            else:
                for _ in range(subject.num_practicals):
                    lessons_to_schedule.append(Lesson(subject, "Практика", group))

        # Shuffle the order of lessons to add randomness
        random.shuffle(lessons_to_schedule)

        # Assign lessons to the schedule
        for lesson in lessons_to_schedule:
            # Find possible lecturers for the lesson
            possible_lecturers = get_possible_lecturers(lesson)
            if not possible_lecturers:
                continue
            lesson.lecturer = random.choice(possible_lecturers)

            # Determine suitable auditoriums based on group size
            if lesson.subgroup:
                students = lesson.group.size // len(lesson.group.subgroups)
            else:
                students = lesson.group.size
            suitable_auditoriums = [
                aud for aud in auditoriums if aud.capacity >= students
            ]
            if not suitable_auditoriums:
                continue
            lesson.auditorium = random.choice(suitable_auditoriums)

            # Try to assign the lesson to both even and odd week timetables
            assigned_odd = assign_randomly(lesson, schedule.odd_timetable)
            assigned_even = assign_randomly(lesson, schedule.even_timetable)

            if not assigned_odd and not assigned_even:
                # If assignment fails, reduce fitness for this schedule
                schedule.fitness = 0

        # Calculate the fitness of the schedule after assigning lessons
        schedule.calculate_fitness()
        population.append(schedule)

    return population


def assign_randomly(lesson, timetable):
    """
    Assigns a lesson to a random available time slot in the given timetable.

    Args:
        lesson (Lesson): The lesson to be scheduled.
        timetable (dict): The timetable to assign the lesson to.

    Returns:
        bool: True if the lesson was successfully assigned, False otherwise.
    """
    assigned = False

    # Shuffle the available time slots to ensure randomness
    available_time_slots = TIME_SLOTS.copy()
    random.shuffle(available_time_slots)

    # Try to assign the lesson to a time slot without conflicts
    for time_slot in available_time_slots:
        if not does_lesson_conflict(lesson, time_slot, timetable):
            lesson.time_slot = time_slot
            timetable[time_slot].append(copy.deepcopy(lesson))
            assigned = True
            break

    return assigned


def selection(population):
    """
    Selects the top-performing schedules (elitism) from the current population.

    Args:
        population (list[TwoWeeksSchedule]): The current generation of schedules.

    Returns:
        list[TwoWeeksSchedule]: The top 20% of schedules based on fitness.
    """
    # Sort the population by fitness in descending order
    population.sort(key=lambda x: x.fitness, reverse=True)
    selected = population[: int(0.2 * len(population))]  # Select top 20%
    return selected


def crossover(parent1, parent2):
    """
    Creates a new schedule (child) by combining lessons from two parent schedules.

    Args:
        parent1 (TwoWeeksSchedule): The first parent schedule.
        parent2 (TwoWeeksSchedule): The second parent schedule.

    Returns:
        TwoWeeksSchedule: The new child schedule with combined lessons.
    """
    child = TwoWeeksSchedule()
    for time_slot in TIME_SLOTS:
        # Randomly decide whether to copy lessons from parent1 or parent2
        if random.random() < 0.5:
            source_lessons_even = parent1.even_timetable[time_slot]
            source_lessons_odd = parent1.odd_timetable[time_slot]
        else:
            source_lessons_even = parent2.even_timetable[time_slot]
            source_lessons_odd = parent2.odd_timetable[time_slot]

        # Copy lessons for the even week
        for lesson in source_lessons_even:
            if not does_lesson_conflict(lesson, time_slot, child.even_timetable):
                child.even_timetable[time_slot].append(copy.deepcopy(lesson))

        # Copy lessons for the odd week
        for lesson in source_lessons_odd:
            if not does_lesson_conflict(lesson, time_slot, child.odd_timetable):
                child.odd_timetable[time_slot].append(copy.deepcopy(lesson))

    # Calculate the fitness of the child schedule
    child.calculate_fitness()
    return child


def mutate(schedule):
    """
    Applies random mutations to a schedule to introduce variability.

    Types of mutations:
    - Transfer lessons between even and odd weeks.
    - Add a new random lesson.
    - Remove an existing lesson.
    - Change the time slot of a lesson.

    Args:
        schedule (TwoWeeksSchedule): The schedule to mutate.
    """
    mutation_rate = 0.1  # 10% chance of mutation for each operation

    for week in ["even", "odd"]:
        timetable = (
            schedule.even_timetable if week == "even" else schedule.odd_timetable
        )
        opposite_timetable = (
            schedule.odd_timetable if week == "even" else schedule.even_timetable
        )

        # Transfer lessons between weeks
        if random.random() < mutation_rate:
            transfer_lesson_between_weeks(timetable, opposite_timetable)

        # Add a new random lesson
        if random.random() < mutation_rate:
            add_random_lesson(timetable)

        # Remove an existing lesson
        if random.random() < mutation_rate:
            remove_random_lesson(timetable)

        # Randomly change the time slot of some lessons
        for time_slot in TIME_SLOTS:
            if timetable[time_slot]:
                for lesson in timetable[time_slot][:]:
                    if random.random() < mutation_rate:
                        original_time_slot = lesson.time_slot
                        new_time_slot = random.choice(TIME_SLOTS)
                        if new_time_slot == original_time_slot:
                            continue
                        if not does_lesson_conflict(lesson, new_time_slot, timetable):
                            timetable[original_time_slot].remove(lesson)
                            lesson.time_slot = new_time_slot
                            timetable[new_time_slot].append(lesson)

    # Recalculate the fitness of the schedule after mutations
    schedule.calculate_fitness()


def transfer_lesson_between_weeks(from_timetable, to_timetable):
    """
    Transfers lessons from one week's timetable to the other, ensuring no conflicts.

    Args:
        from_timetable (dict): The timetable from which lessons are transferred.
        to_timetable (dict): The timetable to which lessons are transferred.
    """
    # Choose a random time slot that contains lessons
    time_slots_with_lessons = [ts for ts in from_timetable if from_timetable[ts]]
    if not time_slots_with_lessons:
        return  # No lessons to transfer if the list is empty

    time_slot = random.choice(time_slots_with_lessons)
    lessons_to_transfer = from_timetable[time_slot][:]

    # Check if lessons can be transferred without conflicts
    can_transfer = True
    for lesson in lessons_to_transfer:
        if does_lesson_conflict(lesson, time_slot, to_timetable):
            can_transfer = False
            break
    if can_transfer:
        # Transfer lessons to the new timetable if no conflict
        from_timetable[time_slot] = []
        to_timetable[time_slot].extend(lessons_to_transfer)


def add_random_lesson(timetable):
    """
    Adds a random lesson to the given timetable, ensuring constraints are met.

    Args:
        timetable (dict): The timetable to which the lesson will be added.
    """
    # Choose a random subject from the subjects list
    subject = random.choice(subjects)
    group = next((g for g in groups if g.number == subject.group_id), None)
    if not group:
        return  # If no group found, skip adding the lesson

    # Choose a random lesson type (Lecture or Practical)
    lesson_type = random.choice(["Лекція", "Практика"])
    lessons_to_add = []

    # Add practical lessons for subgroups if required
    if lesson_type == "Практика" and subject.requires_subgroups and group.subgroups:
        for subgroup in group.subgroups:
            lesson = Lesson(subject, lesson_type, group, subgroup)
            lessons_to_add.append(lesson)
    else:
        lesson = Lesson(subject, lesson_type, group)
        lessons_to_add.append(lesson)

    # Assign lecturer and auditorium for each lesson
    for lesson in lessons_to_add:
        possible_lecturers = get_possible_lecturers(lesson)
        if not possible_lecturers:
            return  # Skip if no suitable lecturer found
        lecturer = random.choice(possible_lecturers)
        lesson.lecturer = lecturer

        # Determine the number of students and find suitable auditoriums
        if lesson.subgroup:
            students = group.size // len(group.subgroups)
            suitable_auditoriums = [
                aud for aud in auditoriums if aud.capacity >= students
            ]
        else:
            students = group.size
            suitable_auditoriums = [
                aud for aud in auditoriums if aud.capacity >= students
            ]

        if not suitable_auditoriums:
            return  # Skip if no suitable auditorium found

        auditorium = random.choice(suitable_auditoriums)
        lesson.auditorium = auditorium

    # Randomly assign time slots to the lessons
    available_time_slots = TIME_SLOTS.copy()
    random.shuffle(available_time_slots)

    for time_slot in available_time_slots:
        conflict = False
        for lesson in lessons_to_add:
            if does_lesson_conflict(lesson, time_slot, timetable):
                conflict = True
                break
        if not conflict:
            # If no conflict, add the lesson to the timetable
            for lesson in lessons_to_add:
                lesson.time_slot = time_slot
                timetable[time_slot].append(copy.deepcopy(lesson))
            break


def remove_random_lesson(timetable):
    """
    Removes a random lesson from the timetable.

    Args:
        timetable (dict): The timetable from which a lesson will be removed.
    """
    # Choose a random lesson from the timetable
    all_lessons = [lesson for lessons in timetable.values() for lesson in lessons]
    if not all_lessons:
        return  # No lessons to remove if the timetable is empty

    lesson_to_remove = random.choice(all_lessons)
    lessons_to_remove = []

    # If it's a lesson with subgroups, remove all related lessons
    if lesson_to_remove.subgroup:
        for lessons in timetable.values():
            for lesson in lessons:
                if (
                    lesson.subject.id == lesson_to_remove.subject.id
                    and lesson.group.number == lesson_to_remove.group.number
                    and lesson.type == lesson_to_remove.type
                    and lesson.subgroup == lesson_to_remove.subgroup
                ):
                    lessons_to_remove.append(lesson)
    else:
        lessons_to_remove.append(lesson_to_remove)

    # Remove the selected lessons from the timetable
    for lesson in lessons_to_remove:
        timetable[lesson.time_slot].remove(lesson)


def run_genetic_algorithm():
    """
    Executes the genetic algorithm to find the optimal schedule.

    The algorithm iterates through multiple generations, selecting, crossover, and mutating
    schedules to optimize the fitness (i.e., minimize conflicts and penalties).

    Returns:
        TwoWeeksSchedule: The best schedule found after the specified number of generations.
    """
    population = create_initial_population()

    for generation in range(GENERATIONS):
        # Select the best schedules based on fitness (elitism)
        selected = selection(population)
        new_population = []

        # Elitism: retain the top 10% of schedules without changes
        elite_size = max(1, int(0.1 * POPULATION_SIZE))
        elites = selected[:elite_size]
        new_population.extend(copy.deepcopy(elites))

        # Perform crossover and mutation for the remaining population
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)

        population = new_population
        best_fitness = max(schedule.fitness for schedule in population)

        # Output progress every 10 generations or if optimal fitness is reached
        if (generation + 1) % 10 == 0 or best_fitness == 1.0:
            print(f"Generation #{generation + 1}. Fitness = {best_fitness}")

        if best_fitness == 1.0:
            print(f"Optimal schedule found at generation {generation + 1}.")
            break

    # Return the best schedule found in the population
    best_schedule = max(population, key=lambda x: x.fitness)
    return best_schedule


def print_schedule(schedule):
    """
    Prints the schedule for both the even and odd weeks in a human-readable format.

    Args:
        schedule (TwoWeeksSchedule): The schedule to print.
    """
    even_week_table = []
    odd_week_table = []
    headers = [
        "Slot",
        "Group",
        "Subject",
        "Type",
        "Lecturer",
        "Auditorium",
        "Number of Students",
        "Auditorium Capacity",
    ]

    # Helper function to format a row for the schedule table
    def create_row(time_slot, lesson):
        timeslot_str = f"{time_slot[0]}, slot #{time_slot[1]}"
        group_str = lesson.group.number
        if lesson.subgroup:
            group_str += f" (Subgroup {lesson.subgroup})"
        subject_str = lesson.subject.name
        type_str = lesson.type
        lecturer_str = lesson.lecturer.name if lesson.lecturer else "N/A"
        auditorium_str = lesson.auditorium.id if lesson.auditorium else "N/A"
        if lesson.subgroup and lesson.group.subgroups:
            students = lesson.group.size // len(lesson.group.subgroups)
        else:
            students = lesson.group.size
        students_str = str(students)
        capacity_str = str(lesson.auditorium.capacity) if lesson.auditorium else "N/A"
        row = [
            timeslot_str,
            group_str,
            subject_str,
            type_str,
            lecturer_str,
            auditorium_str,
            students_str,
            capacity_str,
        ]
        return row

    # Collect and format lessons for the even week
    for time_slot in TIME_SLOTS:
        lessons_even = schedule.even_timetable[time_slot]
        for lesson in lessons_even:
            row = create_row(time_slot, lesson)
            even_week_table.append(row)

    # Collect and format lessons for the odd week
    for time_slot in TIME_SLOTS:
        lessons_odd = schedule.odd_timetable[time_slot]
        for lesson in lessons_odd:
            row = create_row(time_slot, lesson)
            odd_week_table.append(row)

    # Print the even week schedule
    print("\nBest schedule for even week:\n")
    if even_week_table:
        print(
            tabulate(
                even_week_table, headers=headers, tablefmt="github", stralign="center"
            )
        )
    else:
        print("No lessons scheduled for even week.\n")

    # Print the odd week schedule
    print("\nBest schedule for the odd week:")
    if odd_week_table:
        print(
            tabulate(
                odd_week_table, headers=headers, tablefmt="github", stralign="center"
            )
        )
    else:
        print("No lessons scheduled for odd week.\n")


if __name__ == "__main__":
    # Run the genetic algorithm to find the best schedule
    best_schedule = run_genetic_algorithm()
    # Print the final schedule to the console
    print_schedule(best_schedule)
