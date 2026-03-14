*** Settings ***
Documentation     TS_Predict: Verify Predict Cat Breed (EP + State Transition)
...               TC-P01 ~ TC-P07
Resource          resource.txt
Test Teardown     Close Browser

*** Test Cases ***
TC-P01 Predict With Valid Cat Image
    [Documentation]    EP valid: upload cat.jpg → แสดงสายพันธุ์ + confidence + บันทึก history
    Login And Go To Predict
    Upload Cat Image    ${CURDIR}${/}testdata${/}cat.jpg
    Sleep    1s
    Click Predict Button
    Sleep    5s
    Should Show Prediction Result

TC-P02 Predict With Non Cat Image
    [Documentation]    EP invalid: upload dog.jpg → แสดง "ไม่เจอแมว"
    Login And Go To Predict
    Upload Cat Image    ${CURDIR}${/}testdata${/}dog.jpg
    Sleep    1s
    Click Predict Button
    Sleep    5s
    # TODO: ตรวจ "ไม่พบแมวในภาพ" message

TC-P03 Predict Without Image
    [Documentation]    EP invalid: ไม่เลือกรูป → ไม่สามารถกด predict
    Login And Go To Predict
    Sleep    1s
    # Predict button ควร disabled หรือไม่มี
    # TODO: ตรวจ button state

TC-P04 Predict With GPS Granted
    [Documentation]    GPS granted → บันทึก location + map marker
    [Tags]    manual    gps
    Login And Go To Predict
    Upload Cat Image    ${CURDIR}${/}testdata${/}cat.jpg
    Click Predict Button
    Sleep    5s
    # GPS ต้องทดสอบ manual (อนุญาต permission ด้วยมือ)
    # TODO: ตรวจ location data ถูกบันทึก

TC-P05 Predict With GPS Denied
    [Documentation]    GPS denied → แสดง "GPS permission denied"
    [Tags]    manual    gps
    # ต้อง deny GPS permission ด้วยมือ
    Log    Manual test: Deny GPS permission and check message

TC-P06 Predict With GPS Timeout
    [Documentation]    GPS timeout → แสดง "GPS request timeout"
    [Tags]    manual    gps
    Log    Manual test: Block GPS to cause timeout and check message

TC-P07 Predict Image With EXIF Data
    [Documentation]    รูปที่มี EXIF GPS → ใช้ EXIF location แทน GPS
    [Tags]    manual
    Login And Go To Predict
    Upload Cat Image    ${CURDIR}${/}testdata${/}cat_with_exif.jpg
    Click Predict Button
    Sleep    5s
    # TODO: ตรวจ location จาก EXIF ไม่ใช่ GPS ปัจจุบัน

*** Keywords ***
Login And Go To Predict
    Open Browser To Login Page
    Login With Valid Credentials
    Go To    ${PREDICT_URL}
    Sleep    2s
