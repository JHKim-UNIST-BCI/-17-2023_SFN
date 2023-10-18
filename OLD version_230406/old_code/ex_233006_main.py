import time

loop_duration = 1  # in seconds
loop_freq = 60  # in Hz
loop_delay = 1 / loop_freq  # in seconds

start_time = time.monotonic()  # get current time
end_time = start_time + loop_duration  # calculate end time

hello_count = 0  # initialize hello count

while time.monotonic() < end_time:
    # Get start time of loop iteration
    loop_start_time = time.monotonic()

    # Do some computation here
    print("Hello, world!")
    hello_count += 1  # increment hello count

    # Get end time of loop iteration
    loop_end_time = time.monotonic()

    # Calculate time spent in loop iteration
    loop_time = loop_end_time - loop_start_time

    # Subtract loop time from loop delay to maintain desired frequency
    time.sleep(max(0, loop_delay - loop_time))

# Print the hello count after the loop ends
print("Total number of hellos: ", hello_count)
