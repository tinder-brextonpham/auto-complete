import csv
import sys
import talon
from talon import quotations
from talon.signature.bruteforce import extract_signature

def extract_text(message):
	extracted_text = []
	unnecessary_fields = {"Message-ID:", "Date:", "From:", "To:", "Subject:", "Mime-Version:", "Content-Type:", "Content-Transfer-Encoding:", "X-From:", "X-To:", "X-cc:", "X-bcc", "X-Folder", "X-Origin", "X-Filename:"}
	lines = message.split('\n')  
	for line in lines:
		split_line = line.split(" ") 
		if split_line[0] not in unnecessary_fields and not line.isspace():
			extracted_text.append(line)
	return extracted_text
				
def create_filtered_emails_textfile():
	csv.field_size_limit(sys.maxsize)
	data = csv.reader(open('hillary-clinton-emails/Emails.csv'))
	with open('filtered_hillary_emails.txt', 'w') as f:
	    for row in data:
	    	message = row[20]
	    	extracted_text = extract_text(message)
	    	for line in extracted_text:
	        	f.write(str(line) + "\n")

def main():
	# talon.init()
	# with open("filtered_emails.txt") as f:
	# 	lines = f.readlines()	
 #    	for line in lines:
	# 		# reply = quotations.extract_from(line, 'text/plain')
	# 		# print reply
	# 		text, signature = extract_signature(line)
	# 		print signature
	# 		break
	create_filtered_emails_textfile()
	

if __name__ == "__main__":
    main()




