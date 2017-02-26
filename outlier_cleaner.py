#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy
    cleaned_data = []

    ### your code goes here
    errors=numpy.square(predictions-net_worths)
    max_error=numpy.percentile(errors, 90)

    for i in range(len(errors)):
        if errors[i] < max_error:
            cleaned_data.append((ages[i], net_worths[i], errors[i]))
            
        
    return cleaned_data
	

