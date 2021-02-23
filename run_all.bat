cd build
for /L %%i in (0,1,5) Do (
  for /L %%j in (0,1,2) Do (
     3D_object_tracking %%i %%j 0
))
pause
