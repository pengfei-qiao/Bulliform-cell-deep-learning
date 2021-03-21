# Bulliform-cell-deep-learning

This repo includes the scripts used in the study of bulliform cell patterning manuscript





### The attached scripts (median-and-average-with-big-data.py, median-and-average-with-big-data.ipynb) pulls the historical S&P 500 daily prices for the past year, mimics the data streaming process to distribute the data into four partitions, and calculates the mean and median of the prices.

#### The code can be divided into ? parts:

- Part 1:
Pulls the data from https://finance.yahoo.com/quote/%5EGSPC/history/ with the GET method, then stores that to stream in the data as if the data came from a streaming API.

- Part 2:
Partition the data into four (almost if not) equally sized chunks, and store them separately.

- Part 3:
Since the mean of the means from all four lists equals the mean of all four lists combined, I calculated the mean for each partition individually, and calculated the mean of these four means for each column. The results are logged as stdout.

- Part 4:
Since the median of the median from all four lists does **NOT** equal the mean of all four lists combined, I created an algorithm utilizing heap to stream in the data (in this implementation, the data did not get streamed in per se, but it can be easily modified to mimic the streaming process). The results are logged as stdout.
