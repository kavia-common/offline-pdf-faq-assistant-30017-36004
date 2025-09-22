#!/bin/bash
cd /home/kavia/workspace/code-generation/offline-pdf-faq-assistant-30017-36004/faq_chatbot_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

