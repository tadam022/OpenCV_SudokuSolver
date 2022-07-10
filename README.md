This project uses OpenCV and a Convolutional Neural Network to read in an image of a Sudoku Puzzle, solve it and then display the solution 
onto the original image.

The solver assumes that the grid has the lagerst contour in the image, so a grid with a drawn border around it for example, will not be processed correctly.
It also assumes that the puzzle is valid and should only have one possible solution.

Works well if the images are clear, well focused with no heavy shadows. As such this application can easily work on screenshots of digital Sudoku puzzles
from websites for example.

The dataset used for training the CNN was the Chars74k dataset obtained from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

Some of the test images were obtained from: https://icosys.ch/sudoku-dataset
