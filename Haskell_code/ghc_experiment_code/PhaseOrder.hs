{-# LANGUAGE CPP #-}

-- module GHC.Core.Opt.PhaseOrder (getPhaseOrder) where

import qualified Data.HashMap.Strict as HashMap
import Data.List (sortBy)
import Data.Ord (comparing)

{-
1. How this code works: The original Haskell Code will perform runWhen and then place that value into a tuple, where the first value is a key for the optimization performed.
2. This list of tuples will be passed into this file, which will then select a HashMap<String, Int> based on a specific value.
3. Once the HashMap selection is made, we will create a list of size n. The value for our hashmap acts as the index for the order we want the optimization, represented by the key to go.
4. We then return this list back, and the Pipeline will continue as normal.
-}


createHashMap :: HashMap.HashMap String Int
createHashMap = HashMap.fromList []

insertIntoHashMap :: HashMap.HashMap String Int -> String -> Int -> HashMap.HashMap String Int
insertIntoHashMap hashMap key value = HashMap.insert key value hashMap

listToNewList :: [(String, String)] -> HashMap.HashMap String Int -> [(Int, String)]
listToNewList inputList hashMap = map getValue inputList
  where
    getValue (key, opt) = (HashMap.lookupDefault 0 key hashMap, opt)


orderTuples :: [(Int, String)] -> [(Int, String)]
orderTuples inputList = sortBy (comparing fst) inputList

main :: IO ()
main = do

    let my_var = 0

    let my_hashmap = case my_var of
                        0 -> insertIntoHashMap (insertIntoHashMap createHashMap "OptB" 0) "OptA" 1
                        1 -> insertIntoHashMap createHashMap "One" 1
                        2 -> insertIntoHashMap createHashMap "Two" 2
                        3 -> insertIntoHashMap createHashMap "Three" 3
                        4 -> insertIntoHashMap createHashMap "Four" 4

    print my_hashmap

    let list_of_strings = [("OptA", "FunnyMan"), ("OptB", "FunnyWoman")]
    let new_list = listToNewList list_of_strings my_hashmap
    let final_list = orderTuples new_list
    print new_list
    print final_list


