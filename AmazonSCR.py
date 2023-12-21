import pandas as pd

import requests
from bs4 import BeautifulSoup as bs



# url="""
# https://www.amazon.eg/-/en/GAMING-Keyboard-Mouse-PC-Laptop/product-reviews/B09692Z88F/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews
# """

main_url_1="""https://www.amazon.eg/-/en/MFTEK-Rainbow-Gaming-Keyboard-Computer/product-reviews/B085217GYP/ref=cm_cr_getr_d_paging_btm_next_"""

main_url_2="""?ie=UTF8&reviewerType=all_reviews&pageNumber="""

"""
https://www.amazon.eg/-/en/MFTEK-Rainbow-Gaming-Keyboard-Computer/product-reviews/B085217GYP/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1
https://www.amazon.eg/-/en/MFTEK-Rainbow-Gaming-Keyboard-Computer/product-reviews/B085217GYP/ref=cm_cr_getr_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2
https://www.amazon.eg/-/en/MFTEK-Rainbow-Gaming-Keyboard-Computer/product-reviews/B085217GYP/ref=cm_cr_getr_d_paging_btm_next_3?ie=UTF8&reviewerType=all_reviews&pageNumber=3
"""

all_reviews=[]

for pg in range (2,10):
    # pg=1
    url=main_url_1+str(pg)+main_url_2+str(pg)
    print(url)

    response=requests.get(url)


    soup=bs(response.content,'lxml')
    print(soup.prettify())

    # reviews_card=soup.findAll("div",{"class":"a-section celwidget"})
    reviews_card=soup.findAll("div", {"class":"a-section review aok-relative"})

    for card in reviews_card:
        rev_data={}
        rev_header=card.find("a", {"class":"a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold"})
        rev_stars=card.find("span", {"class":"a-icon-alt"}).text[0]
        try:
            rev_title=card.find("span", {"class":"cr-translated-review-content"}).text
        except:
            rev_title=card.find("span", {"class":""}).text
        
        rev_body=card.find("span", {"class":"a-size-base review-text review-text-content"})
        
        try:
            rev_text=rev_body.find("span", {"class":"cr-translated-review-content"}).text
        except:
            rev_text=rev_body.span.text
        
        
        rev_data["title"]=rev_title
        rev_data["text"]=rev_text
        rev_data["stars"]=rev_stars
        if (rev_title!="" and rev_text!=""):
            all_reviews.append(rev_data)
    
reviews_df=pd.DataFrame(all_reviews)
print(reviews_df)
reviews_df.to_csv("reviews.csv", sep=',', encoding="utf-8", index=False, header=False)
    

    
    