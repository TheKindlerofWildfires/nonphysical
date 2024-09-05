/*
    handles special cast
    fills tags=num points with 0
    creates a begin/end array
    counts levels
    maybe computes world bounds
    picks the initial dim
    for each level
        sorts the data (zip with traits)
        updates the tags and maybe the dims
    does a final sort 

    seems to need a lot of sort, probably should go and fix
*/