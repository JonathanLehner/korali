import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
 
    objects = []
    
    # deserialize file
    with (open("states.pickle", "rb")) as openfile:
	    while True:
		    try:
			    objects.append(pickle.load(openfile))
		    except EOFError:
			    break

    print("Number of objects deserialized {}".format(len(objects)))
    last = objects[-1]
    
    # extract states
    location = last['location']
    action = last['action']
    velocity = last['velocity']
    acceleration = last['acceleration']

    # define plotting area
    expand = 1.2
    minx = min(location[0,:])*expand
    maxx = max(location[0,:])*expand
    miny = min(location[1,:])*expand
    maxy = max(location[1,:])*expand

    # get number of states
    _, N = location.shape

    # plot all states
    for i in range(N):
      
      print("Plotting {}/{}".format(i, len(location[0,:])), end='\r')
      plt.figure()
      plt.title("Force {:.1f}, Velocity ({:.1f},{:.1f}), Acceleration ({:.1f},{:.1f})".format(action[i], velocity[0,i], velocity[1,i], acceleration[0,i], acceleration[1,i]))
      plt.plot(location[0,:i],location[1,:i])
      plt.arrow(location[0,i],location[1,i], action[i], 0.)
      plt.xlim((minx, maxx))
      plt.ylim((miny, maxy))
      plt.savefig("figures/states{}.png".format(i))
      plt.close()


