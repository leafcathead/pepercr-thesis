void make10 (int* a) {

    *a = 10;

}

int main () {

    int* a;
    *a = 15;

    if (*a == 15) make10(a);

    return *a;
}