*** Settings ***
Documentation     TS_Admin_User: Verify Admin CRUD User (EP)
...               TC-AU01 ~ TC-AU06
Resource          resource.txt
Test Teardown     Close Browser

*** Test Cases ***
TC-AU01 Add New User
    [Documentation]    EP: เพิ่ม user ใหม่ Name + Username + Password + Role → สำเร็จ
    Login As Admin And Go To Admin
    Click Tab    Users
    Sleep    1s
    Click Add New Button
    Sleep    1s
    # TODO: แก้ selector name ให้ตรงกับ form จริง
    Input Text    name=Name        RobotUser
    Input Text    name=Username    robotuser_test1
    Input Text    name=Password    Robot1234
    # TODO: แก้ selector สำหรับ Role dropdown
    Select From List By Value    name=Role    user
    Click Save Button
    Sleep    2s

TC-AU02 Edit User Keep Password
    [Documentation]    EP: แก้ไข Name, Password ว่าง → password เดิมไม่เปลี่ยน
    Login As Admin And Go To Admin
    Click Tab    Users
    Sleep    2s
    Click Edit On Card
    Sleep    1s
    Clear Element Text    name=Name
    Input Text    name=Name    EditedUser
    # Password ว่าง = ไม่เปลี่ยน
    Click Save Button
    Sleep    2s

TC-AU03 Delete User Confirm OK
    [Documentation]    EP: ลบ user → confirm OK → user หาย
    Login As Admin And Go To Admin
    Click Tab    Users
    Sleep    2s
    Click Delete On Card
    Accept Delete Confirmation
    Sleep    2s

TC-AU04 Add User Role Admin
    [Documentation]    EP: เพิ่ม user ด้วย role=admin → สำเร็จ
    Login As Admin And Go To Admin
    Click Tab    Users
    Sleep    1s
    Click Add New Button
    Sleep    1s
    Input Text    name=Name        AdminUser
    Input Text    name=Username    adminuser_test1
    Input Text    name=Password    Admin1234
    Select From List By Value    name=Role    admin
    Click Save Button
    Sleep    2s

TC-AU05 Current Admin Not In List
    [Documentation]    EP: admin ปัจจุบันไม่ควรอยู่ในรายการ users
    Login As Admin And Go To Admin
    Click Tab    Users
    Sleep    2s
    # TODO: ตรวจว่า admin ที่ login อยู่ไม่แสดงในลิสต์
    Page Should Not Contain    admin

TC-AU06 Normal User Cannot Access Admin
    [Documentation]    EP: role=user → เข้า /admin → redirect /login
    Open Browser To Login Page
    Input Username    ${VALID_USER}
    Input Password    ${VALID_PASSWORD}
    Click Login Button
    Should Be On Main Page
    Go To    ${ADMIN_URL}
    Sleep    3s
    Check Redirected To Login

*** Keywords ***
Login As Admin And Go To Admin
    Open Browser To Login Page
    Login As Admin
    Go To    ${ADMIN_URL}
    Sleep    2s
