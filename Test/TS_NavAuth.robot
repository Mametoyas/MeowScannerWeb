*** Settings ***
Documentation     TS_NavAuth: Verify Navigation & Authorization (State Transition)
...               TC-N01 ~ TC-N10
Resource          resource.txt
Test Teardown     Close Browser

*** Test Cases ***
# ===== ProtectedRoute: ไม่ login =====
TC-N01 Access Main Without Login
    [Documentation]    ไม่ login → /main → redirect /login
    Open Browser To URL    ${MAIN_URL}
    Check Redirected To Login

TC-N02 Access Predict Without Login
    [Documentation]    ไม่ login → /predict → redirect /login
    Open Browser To URL    ${PREDICT_URL}
    Check Redirected To Login

TC-N03 Access Preference Without Login
    [Documentation]    ไม่ login → /preference → redirect /login
    Open Browser To URL    ${PREFERENCE_URL}
    Check Redirected To Login

TC-N04 Access CatMap Without Login
    [Documentation]    ไม่ login → /catmap → redirect /login
    Open Browser To URL    ${CATMAP_URL}
    Check Redirected To Login

TC-N05 Access Admin Without Login
    [Documentation]    ไม่ login → /admin → redirect /login
    Open Browser To URL    ${ADMIN_URL}
    Check Redirected To Login

TC-N06 Access Search Without Login
    [Documentation]    ไม่ login → /search → เข้าถึงได้ปกติ (public)
    Open Browser To URL    ${SEARCH_URL}
    Sleep    2s
    Location Should Contain    /search

# ===== Navigation Links =====
TC-N07 Logout Clears Session
    [Documentation]    Click Logout → ลบ storage → redirect /login
    Open Browser To Login Page
    Login With Valid Credentials
    Click Logout Button
    Sleep    2s
    Check Redirected To Login

TC-N08 Admin User Sees Admin Panel Button
    [Documentation]    role=admin → หน้า /main แสดงปุ่ม "Admin Panel"
    Open Browser To Login Page
    Login As Admin
    Sleep    2s
    # TODO: แก้ selector ให้ตรงกับปุ่ม Admin Panel จริง
    Page Should Contain Element    css=.admin-card

TC-N09 Normal User No Admin Panel Button
    [Documentation]    role=user → หน้า /main ไม่แสดงปุ่ม "Admin Panel"
    Open Browser To Login Page
    Login With Valid Credentials
    Sleep    2s
    Page Should Not Contain Element    css=.admin-card

TC-N10 All Feature Links Work
    [Documentation]    Login แล้ว click ทุก feature → เปิดหน้าถูกต้อง
    Open Browser To Login Page
    Login With Valid Credentials
    # TODO: แก้ selector ให้ตรงกับ link จริงบนหน้า /main
    Go To    ${PREDICT_URL}
    Sleep    2s
    Location Should Contain    /predict
    Go To    ${SEARCH_URL}
    Sleep    2s
    Location Should Contain    /search
    Go To    ${PREFERENCE_URL}
    Sleep    2s
    Location Should Contain    /preference
    Go To    ${CATMAP_URL}
    Sleep    2s
    Location Should Contain    /catmap
