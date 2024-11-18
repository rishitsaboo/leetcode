# Next least greater element on its right
class Solution:
    def findLeastGreater(self, n, arr):
      for i in range(n):
        smallest = None
        for j in range(i+1,n):
          if arr[j] > arr[i]:
            if smallest is None or arr[j] < smallest:
              smallest = arr[j]
        arr[i] = smallest if smallest is not None else -1
      return arr

# Union and Intersection
class Solution:
    def union_and_intersection(self, arr1, arr2, n, m):
      arr1_set = set(arr1) 
      arr2_set = set(arr2) 
      #print(arr1_set,arr2_set)
      union  = arr1_set | arr2_set
      intersection  = arr1_set & arr2_set
      print(" ".join(map(str,sorted(union))))
      print(" ".join(map(str,sorted(intersection))))

# Unique Permutations
from itertools import permutations
class Solution:
    def solve(self, s):
      perm_list = permutations(s)
      unique_perms = set(["".join(p) for p in perm_list])
      sorted_unique_perm = sorted(list(unique_perms))
      for perm in sorted_unique_perm:
        print(perm)
     
# Jet Fighter Captain
#You are in a war. You are handling a F-51 Fighter Jet and have to communicate with your allies. You are waiting for a command to confirm whether to attack or not. 
# There is one issue though, you are not sure if your communication channel has been hacked or not.
#To make sure that the message has been received from a trusted source they will send you an array and an integer 'k'. 
# If the array can be divided into k or more than k segments where every segment's minimum positive value which is not present in the array 
# (for example in arr = {1,3,4,5} that value will be 2 since it is the minimum positive value that is not present in the array) is same, 
# then you have to print "Attack" else print "Wait". Write an algorithm to help the pilot determine what he should do.



# Most Frequent Element
class Solution:
    def MostFrequent(self, arr):
      arr.sort()
      dic = {}
      for i in arr:
        if i in dic:
          dic[i] +=1
        else:
          dic[i] = 1
      max_count = 0
      frequent = None
      for arr,count in dic.items():
        if count > max_count:
          max_count = count
          frequent = arr
      return frequent
    
# Hashmap Implementation
class Solution:
    def map_implementation(self, arr):
      dic = {}
      for i in arr:
        if i in dic:
          dic[i] +=1
        else:
          dic[i] = 1
      return dic

# check-if-n-and-its-double-exist
class Solution:
    def checkIfExist(self, arr: list[int]) -> bool:
        arr.sort()
        n = len(arr)
        for i in range(n):
            double = arr[i] * 2
            left , right = 0 , n-1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == double and mid != i:
                    return 1
                if arr[mid] > double:
                    right = mid - 1
                else:
                    left = mid + 1 
        return 0

#Subarray Sum Equals K
class Solution:
    def subarraySum(self, nums: list[int], k: int) -> int:
        res = 0  
        curr_sum = 0
        presix_sum = {0:1}
        
        for i in nums:
            curr_sum += i
            diff = curr_sum - k
            print(curr_sum , diff)

            res +=  presix_sum.get(diff,0)
            presix_sum[curr_sum] = 1 + presix_sum.get(curr_sum,0)
        return res
        
          
#        class Solution:
#     def winner(self, input_list):
#       dic = {}
#       max_count = 0
#       for i,j in input_list:
#         #print(i[0],i[1])
#         if i in dic:
#           dic[i] += j
#           max_count = j
#           print(max_count)
#         else:
#           dic[i] = j
#       print(dic)
#       max_count = 0
#       frequent = None
#         for i,j in dic.items():
#         if j > max_count:
#           max_count = j
#           frequent = i
#       return frequent

class Solution:
    def winner(self, input_list):
      sum = {}
      index = {}
      for x in range(len(input_list)):
        name = input_list[x][0]
        value = input_list[x][1]
        
        if name not in sum:
          sum[name] = value
        else:
          sum[name] = sum[name] + value

        if name not in index:
          index[name] = [x]
        else:
          index[name].append(x)
      
      print(sum)
      print(index)

      m = 0
      mi = 0
      name = ''
      for i in sum:
        #print(sum[i])
        if m < sum[i]:
          m = sum[i]
          mi = max(index[i])
          name = i
        elif m == sum[i]:
          if mi > max(index[i]):
            m = sum[i]
            mi = max(index[i])
            name = i
      print(m,mi,name)
      
# Divide the Array with Minimum Cost
class Solution:
    def minimumCost(self, nums):
        n = len(nums)
        #nums.sort()
        sum_min_cost = []
        for i in range(1,n-1):
            for j in range(i+1,n):
                sub_arr_1 = nums[:i]
                sub_arr_2 = nums[i:j]
                sub_arr_3 = nums[j:]
        #print(sub_arr_1,sub_arr_2,sub_arr_3)
        
        if len(sub_arr_1) > 0 and len(sub_arr_2) and len(sub_arr_3):
            ans = (sub_arr_1[0]+ sub_arr_2[0]+ sub_arr_3[0])
            sum_min_cost.append(ans)
    #print(sum_min_cost)
        return min(sum_min_cost)  
    
# First Unique Number
class Solution:
    def firstUniqueInteger(self, v):
      n = len(v)
      dic = {}
      index = {}
      for i in range(n):
        if v[i] in dic:
          dic[v[i]] += 1 
        else:
          dic[v[i]] = 1
        if v[i] not in index:
          index[v[i]] = [i]
        else:
          index[v[i]].append(i) 
      #print(dic)
      #print(index)
      res = []
      k = -1
      ind = n
      for key,value in dic.items():
        if (value == 1):
          if index[key][0] < ind:
            ind = index[key][0]
            k = key
      return k
# svaling nodes alternatively in lists
class Node:
  def __init__(self, value):
    self.value = value
    self.next = None

class circularLinkedList:
  def __init__(self):
    self.head = None
    
  def append(self,data):
    new_node = Node(data)
    if not self.head :
      new_node.next = new_node
      self.head = new_node
    else:
      current = self.head
      while current.next != self.head:
        current =  current.next
      current.next = new_node
      new_node.next = self.head
  def printlist(self):
    ind = 0
    current = self.head 
    list_even = []
    list_odd = []
    while current.next != self.head:
      if ind % 2 == 0:
        list_even.append(current.data)
      elif ind % 2 == 1:
        list_odd.append(current.data)
      ind+=1
      current = current.next
    print(list_even)
    print(list_odd)

# Alternating Split of a Circular Linked List
class Solution:
    def split_alternating_nodes(head):
      head1 = head
      head2 = head.next
      i = head1
      j = head2
      ind = 0
      current = i.next.next
      while current != head:
        if ind % 2 == 0:
          i.next = current
          i = i.next
        elif ind % 2 == 1:
          j.next = current
          j = j.next
        current = current.next
        ind += 1
      i.next = head1
      j.next = head2
      return head1,head2
    
# Maximum Gap
class Solution:
    def maximumGap(self, nums):
      nums.sort()
      if len(nums)<2:
        return 0
      else:
        n = len(nums)
        for i in range(n-1):
          res = nums[i+1] - nums[i]
          nums[i] = res
        nums.pop(-1)
        return(max(nums))
#Merge sort
class Solution:
    def merge_sort(self, arr, left, right):
      if len(arr)>1:
        left_arr = arr[:len(arr)//2] 
        right_arr = arr[len(arr)//2:]
        #recursion
        merge_sort(left_arr)
        merge_sort(right_arr)

        #merge
        i = 0 #left arr index
        j = 0 #right arr index
        k = 0 #merged array index
        while i < len(left_arr) and j < len(right_arr):
          if left_arr[arr] < right_arr[arr]:
            arr[k] = left_arr[arr]
            i+=1
          else:
            arr[k] = right_arr[arr]
            j+=1
          k+=1
        
        while i < len(left_arr):
          arr[k] = left_arr[arr]
          i+=1
          k+=1
        while j < len(right_arr):
          arr[k] = left_arr[arr]
          j+=1
          k+=1
# insertion sort 
class Solution:
    def sortColors(self, nums):
      for i in range(1,len(nums)):
        j = i-1
        key = nums[i]
        while key < nums[j] and j>=0:
          nums[j+1] = nums[j]
          j-=1
        nums[j+1] = key
#simple linked list 
class Node:
    def __init__(self, value):
      self.value = value
      self.next = None
class LinkedList:
  
    def __init__(self):
      self.head = None 
      
    def append_Node(self,data):
      new_node = Node(data)
      if not self.head:
          self.head = new_node
      else:
          current = self.head
          while current.next != None:
              current = current.next
          current.next = new_node
       
    def print_linked_list(self):
        current = self.head
        while current.next != None:
            print(current.value)
            current = current.next
        print(current.value)
 # v = [1,9,11,3,2]     
class Solution:
    def solve(self, v, k):
      ls = LinkedList()
      for i in v:
        ls.append_Node(i)
      ls.print_linked_list()

# Tour of all Petrol Pump
class PetrolPump:
    def __init__(self, petrol, distance):
        self.petrol = petrol
        self.distance = distance

class Solution:
    def solve(self, p, n):
      total_petrol = 0
      total_distance = 0
      extra = 0
      first_index = 0
      for i in range(n):
        total_petrol += p[i].petrol 
        total_distance += p[i].distance
        extra += p[i].petrol - p[i].distance
        print(total_petrol,total_distance,extra)
        if extra < 0:
          first_index = i+1
          extra = 0
      if total_petrol >= total_distance:
        return first_index
      else:
        return -1
# First Non Repeating
from collections import deque,defaultdict
class Solution:
    def firstNonRep(self, s):
      char_count = defaultdict(int)
      q = deque()
      out = []
      for i in s:
        char_count[i] += 1
        if char_count[i] == 1:
          q.append(i)
        #print(q,char_count[q[0]]) 
        while (q and char_count[q[0]] > 1):
          q.popleft()
        #print(q,"q")
        if q:
          out.append(q[0])
        else:
          out.append("X")
      return "".join(out)
# previous smaller element
nums = [3,5,6,1,9] 
#expected output[-1,3,5,-1,1]
temp = []
for j in  range(len(nums)):
    i = j-1
    while i >= 0:
        if nums[j] > nums[i]:
            temp.append(nums[i])
            break
        i-=1
    else:
        temp.append(-1)
print(temp)

# calulating area of rectange from histogram 
class Solution:
    def getMaxArea(self, arr, n):
      n = len(arr)
      stack1 = []
      stack2 = []
      temp1 = [-1] * len(arr)
      temp2 = [n] * len(arr)
      max_area = 0
      for i in range(n):
        while stack1 and arr[i] <= arr[stack1[-1]]:
          stack1.pop()
        if stack1:
          temp1[i] = stack1[-1]
        stack1.append(i)
        
      for i in range(n):
        while stack2 and arr[i] < arr[stack2[-1]]:
          index = stack2.pop()
          temp2[index] = i
        stack2.append(i)
      print(temp1)
      print(temp2)
      for i in range(n):
        width = temp2[i] - temp1[i] -1
        area = width * arr[i]
        max_area =max(max_area, area) 
      return(max_area)
# 132 pattern 
class Solution:
    def find132pattern(self, nums):
      stack = []
      cur_min = nums[0]
      for i in nums[1:]:
        while stack and i >= stack[-1][0]:
          stack.pop()
        if stack and i > stack[-1][1]:
          return bool(0)
        stack.append([i,cur_min])
        cur_min = min(cur_min,i)
      return bool(1)
# valid parentheses
class Solution:
    def isValid(self, s: str) -> bool:
        mapper = {")":"(" , "}":"{" , "]":"["}
        stack = []
        for i in s:
            if i in mapper.values():
                stack.append(i)
            elif i in mapper:
                if stack and  stack[-1] == mapper[i]:
                    stack.pop()
                else:
                    return False
        return not stack
1,205,286
#Count the Operations 
# Given a string S consisting of only opening and closing curly brackets '{' and '}', 
# find out the minimum number of operations required to convert the string into a balanced expression.
# A single operation means changing '{' to '}' or vice-versa.
# Return the minimum number of reversals required to balance the bracket sequence. If balancing is not possible, return
class Solution:
    def countop(self, s):
      if len(s) % 2 != 0:
        return -1
      stack = []
      for i in s:
        if i =="{":
          stack.append(i)
        else:
            if stack and stack[-1] == "{":
              stack.pop()
            else:
              stack.append(i)
      #print(stack)
      open_brackets = 0
      closed_brackets = 0
      for i in stack:
        if i == "{":
          open_brackets +=1
        else:
          closed_brackets += 1
      #print(open_brackets,closed_brackets)
      open = open_brackets // 2
      close = closed_brackets //2
      remanaing = 2 *(open_brackets  % 2)
      return (open+close+remanaing)
# Collisions
class Solution:
    def asteroid_collision(self, n, arr):
      stack = []
      for i in arr:
        while stack and i < 0 and stack[-1] > 0:
          if stack[-1] < -i:
            stack.pop()
          elif stack[-1] == -i:
              stack.pop()
              break
          else:
            break
        else:
            stack.append(i)
      return stack
# minimum-number-of-pushes-to-type-word-ii
from collections import Counter
class Solution:
    def minimumPushes(self, word: str) -> int:
        dic = Counter(word)
        count = sorted(dic.values(),reverse = True)
        print(count)
        total = 0
        for i in range(len(count)):
            print(i)
            if (i+1) % 8 == 0:
                total += ((i+1) // 8) * count[i]
            else:
                total += ((i+1) // 8 + 1) * count[i] 
        return total
#integer-to-english-words
class Solution:
    def numberToWords(self, num: int) -> str:
        return self.get_hundreds(num).strip()

    def get_hundreds(self,num):

        if num == 0:
            return "Zero"
        elif num >= 10 ** 9:
            curr = num // (10 ** 9)
            num1 = num - curr * (10 ** 9)
            var2 = self.get_hundreds(curr) + "Billion " +  (self.get_hundreds(num1) if num1 > 0 else "")
            return var2
        elif num >= 1000000 and num <= 999999999:
            curr = num // (10 ** 6)
            num1 = num - curr * (10 ** 6)
            var2 = self.get_hundreds(curr) + "Million " +  (self.get_hundreds(num1) if num1 > 0 else "")
            return var2
        elif num >= 1000 and num <= 999999:
            curr = num // (10 ** 3)
            num1 = num - curr * (10 ** 3)
            var1 = self.get_hundreds(curr)+ "Thousand "+ (self.get_hundreds(num1) if num1 > 0 else "")
            return var1
        elif num < 1000:
            ones = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]
            tens = ["Twenty","Thirty","Forty","Fifty","Sixty","Seventy","Eighty","Ninety"]
            teens = ["Ten","Eleven","Twelve","Thirteen","Fourteen","Fifteen","Sixteen","Seventeen","Eighteen","Nineteen"]
            x = num // 100
            yx = num - x * 100
            y = yx // 10
            z = yx - y * 10
            x_str,y_str ,z_str= "","",""
            if x > 0:
                x_str = ones[x-1] + " Hundred "
            if y > 1 and z !=0:
                y_str = tens[y - 2]+ " "
                z_str = ones[z-1]+ " "
            elif y > 1 and z == 0:
                y_str = tens[y - 2]+ " "
            elif y == 1:
                y_str = teens[z] + " "
            elif y == 0:
                if z != 0:
                    z_str = ones[z-1]+ " "
            return x_str + y_str + z_str

       
         
          
      


          
          

    


        



    




        

        

 

   
          
      
