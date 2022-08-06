This project uses OpenCV and a Convolutional Neural Network to read in an image of a Sudoku Puzzle, solve it and then display the solution 
onto the original image.

Requirements can be found in requirements.txt

The dataset used for training the CNN was the Chars74k computer font characters dataset, obtained from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
The CNN was only trained for the 10 classes corresponding to the numbers 0 to 9 inclusive, namely Samples001 to Samples010 from Eng


The solver assumes that the grid has the largest contour in the image, so a grid with a drawn border around it for example, will not be processed correctly.
It also assumes that the puzzle is valid and should only have one possible solution.

Works well if the images are clear and well defined, with no heavy shadows or a lot of varying illumination. As such this application can easily work on screenshots of digital Sudoku puzzles from websites for example.


Some of the test images were obtained from: https://icosys.ch/sudoku-dataset
