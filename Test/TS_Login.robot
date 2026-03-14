*** Settings ***
Documentation     TS_Login: Verify Login function (Decision Table)
...               TC-L01 ~ TC-L10
Resource          resource.txt
Test Teardown     Close Browser

*** Test Cases ***
# ===== Decision Table: Valid =====
TC-L01 Valid Login
    [Documentation]    valid user + valid password → Redirect /main
    Open Browser To Login Page
    Input Username    ${VALID_USER}
    Input Password    ${VALID_PASSWORD}
    Click Login Button
    Should Be On Main Page

# ===== Decision Table: Invalid Combinations =====
TC-L02 Valid User Invalid Password
    [Documentation]    valid user + wrong password → error message
    Open Browser To Login Page
    Input Username    ${VALID_USER}
    Input Password    ${INVALID_PASSWORD}
    Click Login Button
    Sleep    2s
    Location Should Contain    /login

TC-L03 Invalid User Valid Password
    [Documentation]    wrong user + valid password → error message
    Open Browser To Login Page
    Input Username    ${INVALID_USER}
    Input Password    ${VALID_PASSWORD}
    Click Login Button
    Sleep    2s
    Location Should Contain    /login

TC-L04 Invalid User Invalid Password
    [Documentation]    wrong user + wrong password → error message
    Open Browser To Login Page
    Input Username    ${INVALID_USER}
    Input Password    ${INVALID_PASSWORD}
    Click Login Button
    Sleep    2s
    Location Should Contain    /login

# ===== Decision Table: Empty Fields =====
TC-L05 Empty Username
    [Documentation]    empty user + valid password → HTML5 required validation
    Open Browser To Login Page
    Input Password    ${VALID_PASSWORD}
    Click Login Button
    Sleep    1s
    Location Should Contain    /login

TC-L06 Empty Password
    [Documentation]    valid user + empty password → HTML5 required validation
    Open Browser To Login Page
    Input Username    ${VALID_USER}
    Click Login Button
    Sleep    1s
    Location Should Contain    /login

TC-L07 Both Empty
    [Documentation]    empty user + empty password → HTML5 required validation
    Open Browser To Login Page
    Click Login Button
    Sleep    1s
    Location Should Contain    /login

# ===== Remember Me =====
TC-L08 Login With Remember Me Checked
    [Documentation]    remember me checked → user data in localStorage
    Open Browser To Login Page
    # TODO: แก้ selector ให้ตรงกับ checkbox จริงบนเว็บ
    Select Checkbox    css=input[type="checkbox"]
    Input Username    ${VALID_USER}
    Input Password    ${VALID_PASSWORD}
    Click Login Button
    Should Be On Main Page
    ${user}=    Execute Javascript    return window.localStorage.getItem('user')
    Should Not Be Equal    ${user}    ${None}

TC-L09 Login Without Remember Me
    [Documentation]    remember me unchecked → user data in sessionStorage
    Open Browser To Login Page
    Input Username    ${VALID_USER}
    Input Password    ${VALID_PASSWORD}
    Click Login Button
    Should Be On Main Page
    ${user}=    Execute Javascript    return window.sessionStorage.getItem('user')
    Should Not Be Equal    ${user}    ${None}

# ===== State Transition =====
TC-L10 Already Logged In Redirect
    [Documentation]    logged in → open /login → redirect /main
    Open Browser To Login Page
    Login With Valid Credentials
    Go To    ${LOGIN_URL}
    Sleep    2s
    Should Be On Main Page
