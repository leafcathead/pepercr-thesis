

put_hello :: String -> IO()
put_hello my_name = putStrLn ("Hello, World, " ++ my_name ++ "!")

main :: IO ()
main = do
  put_hello "a"
