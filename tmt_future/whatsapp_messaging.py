#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:37:20 2020

@author: thomas.topilko
"""

from twilio.rest import Client


# Your Account SID from twilio.com/console
account_sid = "ACd1a238d0b4ebd7e7adcef1dadd70b9af"
# Your Auth Token from twilio.com/console
auth_token  = "428e6b1fcca8c32713557e82b3044f33"

client = Client(account_sid, auth_token)

# this is the Twilio sandbox testing number
from_whatsapp_number='whatsapp:+14155238886'
# replace this number with your own WhatsApp Messaging number
to_whatsapp_number='whatsapp:+33768449518'

client.messages.create(body='The code finished running!',
                       from_=from_whatsapp_number,
                       to=to_whatsapp_number)
