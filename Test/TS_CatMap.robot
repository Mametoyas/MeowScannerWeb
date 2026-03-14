*** Settings ***
Documentation     TS_CatMap: Verify Cat Map & Location (EP)
...               TC-CM01 ~ TC-CM04
Resource          resource.txt
Test Teardown     Close Browser

*** Test Cases ***
TC-CM01 CatMap With Location Data
    [Documentation]    EP: user มี location data → แสดง Leaflet map + markers
    Login And Go To CatMap
    Sleep    3s
    # TODO: ตรวจ map element แสดงผล
    Page Should Contain Element    css=.leaflet-container

TC-CM02 CatMap No Location Data
    [Documentation]    EP: user ไม่มี location → "No Cat Locations Yet" + ปุ่มไป /predict
    [Tags]    manual
    # ต้องใช้ user ที่ไม่เคย predict + GPS
    Log    Manual test: Login with user that has no location data

TC-CM03 Click Map Marker
    [Documentation]    EP: click marker → popup แสดง cat name + location ID
    [Tags]    manual
    Login And Go To CatMap
    Sleep    3s
    # TODO: click marker บน leaflet map
    Log    Manual test: Click marker and verify popup content

TC-CM04 Click View In Legend
    [Documentation]    EP: click "View" → เปิด Google Maps tab ใหม่
    [Tags]    manual
    Login And Go To CatMap
    Sleep    3s
    # TODO: click View button ใน legend
    Log    Manual test: Click View and verify Google Maps opens

*** Keywords ***
Login And Go To CatMap
    Open Browser To Login Page
    Login With Valid Credentials
    Go To    ${CATMAP_URL}
    Sleep    2s
