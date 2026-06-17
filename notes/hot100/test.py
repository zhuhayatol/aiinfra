height = [0,1,0,2,1,0,1,3,2,1,2,1]
n = 0
print(len(height))

for i in range(len(height)):
    if i == 0 or i == len(height) - 1:
        continue
    if height[i-1] <= height[i] and height[i+1] <= height[i]:
        continue
    print("now i is ", i)
    lm, rm = max(height[:i]), max(height[i+1:])
    print(i, ":", lm, rm)
    print()