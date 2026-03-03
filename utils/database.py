import streamlit as st
from pymongo import MongoClient
from datetime import datetime
import os

# Get Mongo URI (Cloud first, then local fallback)
if "MONGO_URI" in st.secrets:
    MONGO_URI = st.secrets["MONGO_URI"]
else:
    MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in secrets or environment.")

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client["grocery_tracker"]

# Collections
users = db["users"]
products = db["products"]
deleted_products = db["deleted_products"]
ocr_feedback = db["ocr_feedback"]


def get_user_products(user_email):
    return list(products.find({"user_email": user_email}))


def get_deleted_products(user_email):
    return list(deleted_products.find({"user_email": user_email}))


def add_product(user_email, name, expiry):
    product = {
        "user_email": user_email,
        "name": name,
        "expiry": expiry,
        "added_at": datetime.now()
    }
    products.insert_one(product)


def update_product(product_id, new_name, new_expiry):
    products.update_one(
        {"_id": product_id},
        {"$set": {"name": new_name, "expiry": new_expiry}}
    )


def delete_product(product_id):
    product = products.find_one({"_id": product_id})
    if product:
        deleted_products.insert_one(product)
        products.delete_one({"_id": product_id})


def restore_product(product_id):
    product = deleted_products.find_one({"_id": product_id})
    if product:
        products.insert_one(product)
        deleted_products.delete_one({"_id": product_id})


def get_ocr_feedback(user_email, image_path):
    feedback = ocr_feedback.find_one(
        {"user_email": user_email, "image_path": image_path}
    )
    if feedback:
        return feedback["user_product"], feedback["user_expiry"]
    return None, None


def upsert_ocr_feedback(image_path, user_email, pred_product, pred_expiry, user_product, user_expiry):
    feedback = {
        "user_email": user_email,
        "image_path": image_path,
        "predicted_product": pred_product,
        "predicted_expiry": pred_expiry,
        "user_product": user_product,
        "user_expiry": user_expiry,
        "timestamp": datetime.now()
    }
    ocr_feedback.update_one(
        {"user_email": user_email, "image_path": image_path},
        {"$set": feedback},
        upsert=True
    )