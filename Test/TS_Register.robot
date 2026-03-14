*** Settings ***
Documentation     TS_Register: Verify Register & Password Validation (BVA + EP)
...               TC-R01 ~ TC-R17
Resource          resource.txt
Test Teardown     Close Browser

*** Test Cases ***
# ===== EP Valid: สมัครสำเร็จ =====
TC-R01 Register With All Valid Data
    [Documentation]    EP valid: กรอกครบทุกช่อง → สมัครสำเร็จ redirect /main
    Open Browser To Register Page
    Input Register Name    TestUser
    Input Register Username    testuser_robot1
    Input Register Password    Test1234
    Input Register Confirm Password    Test1234
    Click Register Button
    Sleep    3s
    # TODO: ตรวจ redirect /main หรือ success message

# ===== BVA: Password Length =====
TC-R02 Password 7 Chars BVA Min Minus 1
    [Documentation]    BVA: 7 ตัว < ขอบเขตล่าง 8 → error
    Open Browser To Register Page
    Input Register Password    Test123
    Sleep    1s
    Password Should Show Error

TC-R03 Password 8 Chars BVA Min
    [Documentation]    BVA: 8 ตัว = ขอบเขตล่าง → ผ่าน length check
    Open Browser To Register Page
    Input Register Password    Test1234
    Sleep    1s
    Password Should Show Valid

TC-R04 Password 9 Chars BVA Min Plus 1
    [Documentation]    BVA: 9 ตัว = ขอบเขตล่าง+1 → ผ่าน length check
    Open Browser To Register Page
    Input Register Password    Test12345
    Sleep    1s
    Password Should Show Valid

# ===== EP Invalid: ไม่มีตัวเลข / ไม่มีตัวอักษร / อักขระพิเศษ =====
TC-R05 Password Letters Only No Number
    [Documentation]    EP: "Testtest" ไม่มีเลข → error
    Open Browser To Register Page
    Input Register Password    Testtest
    Sleep    1s
    Password Should Show Error

TC-R06 Password Numbers Only No Letter
    [Documentation]    EP: "12345679" ไม่มีอักษร → error
    Open Browser To Register Page
    Input Register Password    12345679
    Sleep    1s
    Password Should Show Error

TC-R07 Password With Special Characters
    [Documentation]    EP: "Test@123!" มีอักขระพิเศษ → error
    Open Browser To Register Page
    Input Register Password    Test@123!
    Sleep    1s
    Password Should Show Error

TC-R08 Password Same Char Repeated
    [Documentation]    EP: "aaaaaaaa" ซ้ำ 8 ตัว → error
    Open Browser To Register Page
    Input Register Password    aaaaaaaa
    Sleep    1s
    Password Should Show Error

# ===== EP Invalid: Simple Patterns =====
TC-R09 Pattern Sequential Numbers
    [Documentation]    EP: "12345678" → error ง่ายเกินไป
    Open Browser To Register Page
    Input Register Password    12345678
    Sleep    1s
    Password Should Show Error

TC-R10 Pattern Sequential Letters
    [Documentation]    EP: "abcdefgh" → error ง่ายเกินไป
    Open Browser To Register Page
    Input Register Password    abcdefgh
    Sleep    1s
    Password Should Show Error

TC-R11 Pattern Keyboard
    [Documentation]    EP: "qwertyui" → error ง่ายเกินไป
    Open Browser To Register Page
    Input Register Password    qwertyui
    Sleep    1s
    Password Should Show Error

TC-R12 Pattern Common Word
    [Documentation]    EP: "password1" → error ง่ายเกินไป
    Open Browser To Register Page
    Input Register Password    password1
    Sleep    1s
    Password Should Show Error

# ===== Confirm Password =====
TC-R13 Confirm Password Mismatch
    [Documentation]    confirm != password → error + ปุ่ม disabled
    Open Browser To Register Page
    Input Register Password    Test1234
    Input Register Confirm Password    Test5678
    Sleep    1s
    Confirm Password Should Show Mismatch

# ===== Required Fields =====
TC-R14 Empty Name Field
    [Documentation]    name ว่าง → required validation
    Open Browser To Register Page
    Input Register Username    testuser2
    Input Register Password    Test1234
    Input Register Confirm Password    Test1234
    Click Register Button
    Sleep    1s
    Location Should Contain    /register

TC-R15 Empty Username Field
    [Documentation]    username ว่าง → required validation
    Open Browser To Register Page
    Input Register Name    TestUser
    Input Register Password    Test1234
    Input Register Confirm Password    Test1234
    Click Register Button
    Sleep    1s
    Location Should Contain    /register

# ===== Duplicate Username =====
TC-R16 Duplicate Username
    [Documentation]    username ซ้ำ → API error
    Open Browser To Register Page
    Input Register Name    TestUser
    Input Register Username    demo
    Input Register Password    Test1234
    Input Register Confirm Password    Test1234
    Click Register Button
    Sleep    2s
    # TODO: ตรวจ error message "username already exists"

# ===== State Transition =====
TC-R17 Already Logged In Redirect
    [Documentation]    logged in → open /register → redirect /main
    Open Browser To Login Page
    Login With Valid Credentials
    Go To    ${REGISTER_URL}
    Sleep    2s
    Should Be On Main Page
