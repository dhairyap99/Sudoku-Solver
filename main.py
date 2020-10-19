import argparse
import detection
import cv2

# create parser
parser = argparse.ArgumentParser()

# add arguments to the parser
parser.add_argument("language")

# parse the arguments
args = parser.parse_args()

question = cv2.imread(args.language)
question = cv2.resize(question, (381,381))

image = detection.run_detection(args.language)

cv2.imshow("questions", question)
cv2.imshow("answer", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#python main.py "images/sudoku5.jpeg"