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
          print(stack1,arr[i],arr[stack1[-1]])
          stack1.pop()
          print(stack1.pop())
        if stack1:
          temp1[i] = stack1[-1]
        stack1.append(i)
        
      for i in range(n):
        while stack2 and arr[i] > arr[stack2[-1]]:
          index = stack2.pop()
          temp2[index] = i
        stack2.append(i)
      print(temp1)
        # print(temp2)
      for i in range(n):
        width = temp2[i] - temp1[i] -1
        area = width * arr[i]
        max_area =max(max_area, area) 
      return(max_area)

          
# Palindromic linked list
'''class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None'''
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
      if head is None or head.next is None:
        return True
      first = self.end_of_first_half(head)
      second = self.reverse_list(first.next)
      result = True
      f_pos = head
      s_pos = second
      while result and s_pos is not None:
        if f_pos.val != s_pos.val:
          result = False
        f_pos = f_pos.next
        s_pos = s_pos.next
      first.next = self.reverse_list(second)
      return result
          
    def end_of_first_half(self, head):
      slow = head
      fast = head
      while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
      return slow
# REVERSEING THE LIST
    def reverse_list(self, head):
      prev = None
      curr = head
      while curr:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
      return prev
##insertion at beginning, middle, end
'''class ListNode:
    def __init__(self, x):
        self.val = x
        self.prev = None
        self.next = None'''
class Solution:
    def addVal(self, head, val):
        head = self.insert_beginning(head, val)
        head = self.insert_middle(head, val)
        head = self.insert_end(head, val)
        return head
    def insert_beginning(self, head, val):
      new_node = ListNode(val)
      new_node.next = head
      if head:
        head.prev = new_node
      return new_node
      
    def insert_middle(self, head, val):
      count = 0
      tail = head
      while tail:
        tail = tail.next
        count += 1
      if count % 2 == 0:
        dummy = ListNode(0)
        dummy.next = head
        slow = dummy
        fast = head
        while fast and fast.next:
          slow = slow.next
          fast = fast.next.next
        new_node = ListNode(val)
        temp = slow.next
        slow.next = new_node
        new_node.prev = slow
        new_node.next = temp
        if temp:
          temp.prev = new_node
        return head
      else:
        slow = head
        fast = head
        while fast and fast.next:
          slow = slow.next
          fast = fast.next.next
        new_node = ListNode(val)
        temp = slow.next
        slow.next = new_node
        new_node.prev = slow
        new_node.next = temp
        if temp:
          temp.prev = new_node
        return head
      
    def insert_end(self, head, val):
      new_node = ListNode(val)
      if head is None:
        head = new_node
      else:
        pointer = head
        while pointer.next is not None:
          pointer = pointer.next
        pointer.next = new_node
        new_node.prev = pointer
      return head
        
## top-k-frequent-elements 
## very important
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}
        freq = [[] for i in range(len(nums) + 1)]

        for n in nums:
            count[n] = 1 + count.get(n,0)
        for n , c in count.items():
            freq[c].append(n)
        
        res = []
        for i in range(len(freq)-1,0,-1):
            for n in freq[i]:
                res.append(n)
                if len(res) == k:
                    return res
                
## count-number-of-maximum-bitwise-or-subsets

class Solution:
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        max_or = 0
        for i in nums:
            max_or |= i 
        def subseor(i,curr_or):
            if i == len(nums):
                return 1 if max_or == curr_or else 0
            include = subseor(i+1,curr_or | nums[i])
            exclude = subseor(i+1,curr_or)

            return include + exclude
        
        return subseor(0,0)
## Concatenate the Nodes
class Solution:
  def concatenate(self, root):
    arr = []
    def helper(node):
      if node is None:
        return 
      arr.append(node.data)
      helper(node.left)
      helper(node.right)
    helper(root)
    if len(arr) == 0:
      return 0
    class largest(str):
      def __lt__ (self,other):
        return self+ other > other+ self
    heap = []
    for i in arr:
      heapq.heappush(heap,largest(str(i)))
    result = []
    while heap:
      result.append(heapq.heappop(heap))
    largest_num = "".join(result)
    print("0" if largest_num[0] == "0" else largest_num)
## Three Numbers Product - Input is a BST
class Solution:
    def findTarget(self, root, k):
      def inorder(node):
        if not node:
          return []
        return inorder(node.left) + [node.val] + inorder(node.right)

      elements = inorder(root)
      n = len(elements)
      for i in range(n):
        target = k // elements[i]
        left, right = i+1, n-1
        while left < right:
          product = elements[left] * elements[right]
          if product == target:
            return True
          elif product < target:
            left +=1
          else:
            right -=1
      return False
