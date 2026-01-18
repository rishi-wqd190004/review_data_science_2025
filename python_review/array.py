# two sum
class TwoSum:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        diff = 0
        nums_in_idx = {}
        for idx, val in enumerate(nums):
            diff = target - val
            print(nums_in_idx)
            if diff in nums_in_idx:
                return [nums_in_idx[diff], idx]
            nums_in_idx[val] = idx

op = TwoSum()
res = op.twoSum(nums=[2,7,11,15], target=9)
#print(res)

# two arrays merged median

## Option 1:
import statistics

class TwoArrayMedian():
    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
        merged_arr = nums1.extend(nums2)
        merged_arr = sorted(merged_arr)
        return statistics.median(merged_arr)
    
## Option 2:
class TwoArrayMedian():
    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
        merged_arr = nums1 + nums2
        merged_arr = sorted(merged_arr)
        n = len(merged_arr)
        middle_idx = n // 2
        if n % 2 == 0:
            median_val = (merged_arr[middle_idx - 1] + merged_arr[middle_idx]) / 2
        else:
            median_val = merged_arr[middle_idx]
        return median_val