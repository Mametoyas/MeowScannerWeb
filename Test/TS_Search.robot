*** Settings ***
Documentation     TS_Search: Verify Search Cat function (EP)
...               TC-S01 ~ TC-S07
Resource          resource.txt
Test Setup        Open Browser To Search Page
Test Teardown     Close Browser

*** Test Cases ***
TC-S01 Search By CatName
    [Documentation]    EP valid: พิมพ์ "Siamese" → แสดงแมวที่ CatName มี "Siamese"
    Sleep    2s
    Input Search Keyword    Siamese
    Sleep    1s
    Should Show Cat Results

TC-S02 Search By CatPersonal
    [Documentation]    EP valid: พิมพ์คำจาก personality
    Sleep    2s
    # TODO: แก้ keyword ให้ตรงกับข้อมูลจริงในระบบ
    Input Search Keyword    friendly
    Sleep    1s
    Should Show Cat Results

TC-S03 Search By CatDetails
    [Documentation]    EP valid: พิมพ์คำจาก details
    Sleep    2s
    Input Search Keyword    Thailand
    Sleep    1s
    Should Show Cat Results

TC-S04 Search Empty Shows All
    [Documentation]    EP: ลบคำค้นให้ว่าง → แสดงแมวทั้งหมด
    Sleep    2s
    Input Search Keyword    test
    Sleep    1s
    Clear Search
    Sleep    1s
    Should Show Cat Results

TC-S05 Search No Match
    [Documentation]    EP invalid: keyword ไม่มีในระบบ → "No cats found"
    Sleep    2s
    Input Search Keyword    xxxNoMatch999
    Sleep    1s
    Should Show No Results

TC-S06 Search Case Insensitive
    [Documentation]    EP: "siamese" (ตัวเล็ก) → ผลเหมือน "Siamese"
    Sleep    2s
    Input Search Keyword    siamese
    Sleep    1s
    # TODO: ตรวจผลเหมือน TC-S01

TC-S07 Search Partial Match
    [Documentation]    EP: พิมพ์บางส่วน "Si" → แสดงแมวที่มี "Si"
    Sleep    2s
    Input Search Keyword    Si
    Sleep    1s
    Should Show Cat Results
