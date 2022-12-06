from django import forms
from django.contrib.auth.forms import ReadOnlyPasswordHashField
from django.contrib.auth.forms import PasswordResetForm 
from .models import User
from django.contrib.auth.forms import SetPasswordForm
from django.contrib.auth import (
    authenticate, get_user_model, password_validation,
)

class UserCreationForm(forms.ModelForm):
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput)
    password2 = forms.CharField(
        label='Password confirmation', widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ('email', 'username')

    def clean_password2(self):
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords don't match")
        return password2

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password1"])
        if commit:
            user.save()
        return user


class SetPasswordFormCustom(SetPasswordForm):
    new_password1 = forms.CharField(
        label="New password",
        widget=forms.PasswordInput(attrs={'class':'form-control','placeholder':'Password'}),
        strip=False,
    )
    new_password2 = forms.CharField(
        label="Confirm new password",
        strip=False,
        widget=forms.PasswordInput(attrs={'class':'form-control','placeholder':'Confirm password'}),
        help_text=password_validation.password_validators_help_text_html(),
    )

class UserChangeForm(forms.ModelForm):
    password = ReadOnlyPasswordHashField()

    class Meta:
        model = User
        fields = ('email', 'password', 'username','is_active', 'is_admin')

    def clean_password(self):
        return self.initial["password"]

class PasswordResetForm(PasswordResetForm):
    def get_users(self, email):
        users = tuple(super().get_users(email))
        if users:
            return users
        msg = '"{email}" was not found in our system.'
        raise forms.ValidationError({'email': msg.format(email=email)})


class AccountForm(forms.ModelForm):
    # パスワード入力：非表示対応
    password = forms.CharField(widget=forms.PasswordInput(),label="パスワード")

    class Meta():
        # ユーザー認証
        model = User
        # フィールド指定
        fields = ('username','email','password')
        # フィールド名指定
        labels = {'username':"ユーザー名",'email':"メール"}