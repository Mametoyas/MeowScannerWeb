*** Settings ***
Documentation     TS_MainStats: Verify Main Page Stats (EP)
...               TC-M01 ~ TC-M02
Resource          resource.txt
Test Teardown     Close Browser

*** Test Cases ***
TC-M01 New User Shows Discovery Message
    [Documentation]    EP: user ใหม่ (0 predictions) → "Start your cat discovery journey"
    [Tags]    manual
    # ต้องใช้ user ที่เพิ่งสมัครใหม่
    Open Browser To Login Page
    # TODO: ใช้ credentials ของ user ใหม่
    Login With Valid Credentials
    Sleep    2s
    # TODO: ตรวจ message "Start your cat discovery journey"

TC-M02 Active User Shows Stats
    [Documentation]    EP: user ที่มี history → แสดง predictions_made, cats_discovered, locations_mapped
    Open Browser To Login Page
    Login With Valid Credentials
    Sleep    2s
    # TODO: ตรวจ stats elements แสดงตัวเลข
    # Page Should Contain Element    css=.stat-card

TC-M03 Welcome Message Shows User Name
    [Documentation]    EP: แสดง "Welcome, {name}" ตรงกับ user ที่ login
    Open Browser To Login Page
    Login With Valid Credentials
    Sleep    2s
    # TODO: ตรวจ Welcome message มีชื่อ user
