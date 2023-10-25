setTo10 :: Integer -> IO Integer
setTo10 n = if n <= 10 then return(n) else setTo10(n-1)

main :: IO()
main = do
    result <- setTo10 11
    print result