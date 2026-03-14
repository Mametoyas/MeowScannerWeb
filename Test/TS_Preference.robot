*** Settings ***
Documentation     TS_Preference: Verify Cat Recommendation Quiz (Decision Table)
...               TC-PF01 ~ TC-PF07
Resource          resource.txt
Test Teardown     Close Browser

*** Test Cases ***
# ===== Decision Table: ครบทุกข้อ =====
TC-PF01 Quiz Combo A-B-C
    [Documentation]    คอนโด + เวลาน้อย + ติดหนึบ → แสดงแมวแนะนำ
    Login And Go To Preference
    Select Radio Option    location      A
    Select Radio Option    freeTime      B
    Select Radio Option    personality   C
    Click Submit Quiz
    Sleep    3s
    Should Show Recommendation

TC-PF02 Quiz Combo B-A-A
    [Documentation]    บ้าน + มีเวลา + นิ่งๆ → แสดงแมวแนะนำ (ผลต่างจาก PF01)
    Login And Go To Preference
    Select Radio Option    location      B
    Select Radio Option    freeTime      A
    Select Radio Option    personality   A
    Click Submit Quiz
    Sleep    3s
    Should Show Recommendation

TC-PF03 Quiz Combo C-C-B
    [Documentation]    บ้านสวน + WFH + ซน → แสดงแมวแนะนำ
    Login And Go To Preference
    Select Radio Option    location      C
    Select Radio Option    freeTime      C
    Select Radio Option    personality   B
    Click Submit Quiz
    Sleep    3s
    Should Show Recommendation

# ===== EP: ไม่เลือก =====
TC-PF04 Quiz Submit No Selection
    [Documentation]    ไม่เลือกข้อใดเลย → required validation
    Login And Go To Preference
    Click Submit Quiz
    Sleep    1s
    Location Should Contain    /preference

TC-PF05 Quiz Submit Partial Selection
    [Documentation]    เลือกแค่ข้อ 1 → required validation สำหรับข้อที่ขาด
    Login And Go To Preference
    Select Radio Option    location    A
    Click Submit Quiz
    Sleep    1s
    Location Should Contain    /preference

# ===== State Transition: ปุ่มหลังผลลัพธ์ =====
TC-PF06 Retry Quiz After Result
    [Documentation]    "ทำแบบทดสอบใหม่" → reset quiz กลับหน้าคำถาม
    Login And Go To Preference
    Select Radio Option    location      A
    Select Radio Option    freeTime      A
    Select Radio Option    personality   A
    Click Submit Quiz
    Sleep    3s
    Should Show Recommendation
    Click Retry Quiz
    Sleep    1s
    # TODO: ตรวจว่ากลับหน้าคำถาม

TC-PF07 Go To Predict After Result
    [Documentation]    "ไปสแกนแมว" → redirect /predict
    Login And Go To Preference
    Select Radio Option    location      A
    Select Radio Option    freeTime      A
    Select Radio Option    personality   A
    Click Submit Quiz
    Sleep    3s
    Should Show Recommendation
    # TODO: แก้ selector ให้ตรง
    Click Element    css=.scan-btn
    Sleep    2s
    Location Should Contain    /predict

*** Keywords ***
Login And Go To Preference
    Open Browser To Login Page
    Login With Valid Credentials
    Go To    ${PREFERENCE_URL}
    Sleep    2s
