import datetime


class Logger():
	def __init__(self, output_file):
		self.output_file = output_file
		open(self.output_file, "w").close()  # create logging file

	def log(self, message):
		timestamp = datetime.datetime.now()
		output = f"{timestamp}\t" + message
		with open(self.output_file, "a") as f:
			f.write(output + "\n")  # write to logging file
