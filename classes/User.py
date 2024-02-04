import math;

class User(object):

	# Initial value for id number
	# For each user add increment the id by 1
	count = 1000;

	def __init__(self, user_name, age, gender, height, weight, weight_goal, activity_level):
		self.user_id = User.count;
		self.user_name = user_name;
		self.age = age;
		self.gender = gender;
		self.height = height;
		self.weight = weight;
		self.weight_goal = weight_goal;
		self.activity_level = activity_level;
		User.count += 1;

	def printUserDetails(self):
		# Print all user details
		if self.gender == 'm':
			gender = "Male";
		else:
			gender = "Female";
		
		if self.weight_goal =='loss':
			goal = 'Loss Weight'
		elif self.weight_goal =='gain':
			goal = 'Gain Weight'
		else:
			goal = 'Maintain Weight'

		if self.activity_level == 1:
			activity = "Sedentary"
		elif self.activity_level == 2:
			activity ="Lightly Active"
		elif self.activity_level == 3:
			activity ="Moderately Active"
		elif self.activity_level == 4:
			activity ="Very Active"
		elif self.activity_level == 5:
			activity ="Extra Active"

		return "Name           : " + self.user_name + "<br>"+ "Age            : " + str(self.age) + "<br>"+"Gender         : " + gender + "<br>"+ "Height         : " + str(self.height) + " cm<br>"+ "Weight         : " + str(self.weight) + " kg<br>"+ "Weight Goal    : " + str(goal) + " <br>"+ "Activity Level : " + str(activity) + "<br>"

	def getBMI(self):
		# Calculate body mass index, bmi to get the weight status
		height = self.height/100; # Convert cm to m
		return self.weight/(height*height);

	def getBMR(self):
		# Calculate bmr based on gender, else return 0(error)
		if self.gender == 'm':
			return math.ceil(66 + (13.7 * self.weight) + (5.0 * self.height) - (6.8 * self.age));
		elif self.gender == 'f':
			return math.ceil(655 + (9.6 * self.weight) + (1.8 * self.height) - (4.7 * self.age));
		else:
			return 0;

	def getDailyCalories(self):
		bmr = self.getBMR();
		# Calculate total calories per day based on the activity_level selected
		# Each activity level has specific activity factor for calories calculation
		if self.activity_level == 1: # little exercise, desk job
			return math.ceil(bmr * 1.2);
		elif self.activity_level == 2: # light exercise, sports 1-3 days/week
			return math.ceil(bmr * 1.375);
		elif self.activity_level == 3: # moderate exercise, sports 3-5 days/week
			return math.ceil(bmr * 1.55);
		elif self.activity_level == 4: # hard exercise, sports 6-7 days/week
			return math.ceil(bmr * 1.725);
		elif self.activity_level == 5: # hard daily exercise, sports + physical labor job or 2 times training per day
			return math.ceil(bmr * 1.9);
		else:
			return 0;

	def getBMIStatus(self):
		bmi = self.getBMI();
		# Determine the bmi status
		if bmi < 18.5: return "Underweight";
		elif bmi >= 18.5 and bmi < 25.0: return "Normal";
		elif bmi >= 25.0 and bmi < 30.0: return "Overweight";
		else: return "Obese";
		"""
		Below 18.5	Underweight
		18.5 – 25.0	Normal
		25.0 – 30.0	Overweight
		30.0 – 35.0	Class 1 Obese
		35.0 – 40.0	Class 2 Obese
		Above 40.0	Class 3 Obese

		"""
		
	def getCaloriesChangeRequired(self):

		tdee = self.getDailyCalories()

		if self.weight_goal == 'loss':
			caloric_goal = tdee - 500
		elif self.weight_goal == 'gain':
			caloric_goal = tdee + 500
		else:
			caloric_goal = tdee
		
		return caloric_goal

