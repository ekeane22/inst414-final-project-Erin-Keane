'''
To process the data: 

I have to start with the flight data. There are over 100 columns and 80% 
are useless to me. I have to delete those and filter out an even amount of 
flights per airport. I have to do this 12 times for all different 12 csv's. 
Then, they can be downlaoded and concatonated in etl. 

As far as the weather data, it has 8 million rows. Not sure what to do about that.
I plan to filter out all data from 2016 - 2021 so it leaves 2022 data. 
The collection process will probably fail data standards but itll give me 
enough rows/data that it might be a decent estimate. I'll probably have to 
filter it anyway so its under 1 million rows. 

'''