@echo off
SET THEFILE=C:\LazarusProject\WesChat\WesChat.exe
echo Linking %THEFILE%
C:\lazarus\fpc\3.2.2\bin\x86_64-win64\ld.exe -b pei-x86-64  --gc-sections    --entry=_mainCRTStartup    -o C:\LazarusProject\WesChat\WesChat.exe C:\LazarusProject\WesChat\link27656.res
if errorlevel 1 goto linkend
goto end
:asmend
echo An error occurred while assembling %THEFILE%
goto end
:linkend
echo An error occurred while linking %THEFILE%
:end
