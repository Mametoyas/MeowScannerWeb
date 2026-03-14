*** Settings ***
Documentation     TS_Admin_Cat: Verify Admin CRUD Cat (EP)
...               TC-A01 ~ TC-A06
Resource          resource.txt
Test Teardown     Close Browser

*** Test Cases ***
TC-A01 Add New Cat
    [Documentation]    EP: เพิ่มแมวใหม่ CatName + CatPersonal + Cat → สำเร็จ
    Login As Admin And Go To Admin
    Click Tab    Cats
    Sleep    1s
    Click Add New Button
    Sleep    1s
    # TODO: แก้ selector name ให้ตรงกับ form จริง
    Input Text    name=CatName       TestCatRobot
    Input Text    name=CatPersonal   Friendly and playful
    Input Text    name=Cat           A test cat for automation
    Click Save Button
    Sleep    2s

TC-A02 Edit Existing Cat
    [Documentation]    EP: แก้ไข CatName → อัปเดตสำเร็จ CatID ไม่เปลี่ยน
    Login As Admin And Go To Admin
    Click Tab    Cats
    Sleep    2s
    Click Edit On Card
    Sleep    1s
    Clear Element Text    name=CatName
    Input Text    name=CatName    UpdatedCatName
    Click Save Button
    Sleep    2s

TC-A03 Delete Cat Confirm OK
    [Documentation]    EP: ลบแมว → confirm OK → แมวหายจากรายการ
    Login As Admin And Go To Admin
    Click Tab    Cats
    Sleep    2s
    Click Delete On Card
    Accept Delete Confirmation
    Sleep    2s

TC-A04 Delete Cat Cancel
    [Documentation]    EP: ลบแมว → Cancel → แมวยังอยู่
    Login As Admin And Go To Admin
    Click Tab    Cats
    Sleep    2s
    Click Delete On Card
    Dismiss Delete Confirmation
    Sleep    1s

TC-A05 Add Cat Empty Name
    [Documentation]    EP invalid: CatName ว่าง → required validation
    Login As Admin And Go To Admin
    Click Tab    Cats
    Sleep    1s
    Click Add New Button
    Sleep    1s
    Input Text    name=CatPersonal    Test
    Input Text    name=Cat            Test
    Click Save Button
    Sleep    1s
    # TODO: ตรวจ validation message

TC-A06 Search Filter Cats
    [Documentation]    EP: พิมพ์ค้นหาใน search → กรองตาม CatName/ID/Personal
    Login As Admin And Go To Admin
    Click Tab    Cats
    Sleep    2s
    # TODO: แก้ selector ให้ตรง
    Input Text    css=.search-input    Bengal
    Sleep    1s

*** Keywords ***
Login As Admin And Go To Admin
    Open Browser To Login Page
    Login As Admin
    Go To    ${ADMIN_URL}
    Sleep    2s
